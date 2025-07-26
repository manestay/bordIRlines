import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import List

import torch
from data_lib import (
    LLM_CHOICES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_UN_PEACEKEEPER,
)
from generation_lib import (
    ModelDetails,
    clean_query_dict,
    get_model_details,
    get_multiple_choice,
    get_other_lang,
    get_user_prompt,
    load_bordirlines,
    load_json_files,
    match_one_letter,
)
from openai import OpenAI
from tqdm import tqdm
from transformers import set_seed

DEFAULT_SEED = 4321

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load and process retrieval data from JSON files.")
parser.add_argument(
    "--retrieval_over",
    "-m",
    type=str,
    choices=["no_ir", "qlang", "qlang_en", "rel_langs", "en", "ablate_swap"],
    default="qlang",
    help="Retrieval mode to use",
)
parser.add_argument(
    "--num_paragraphs",
    "-n",
    type=int,
    default=10,
    help="Number of paragraphs to retrieve",
)
parser.add_argument(
    "--llm", "-l", type=str, choices=LLM_CHOICES, default="gpt-4", help="LLM to use for generation"
)
parser.add_argument(
    "--role",
    "-r",
    type=str,
    choices=["vanilla", "UN"],
    default="vanilla",
)
parser.add_argument("--query_path", "-q", type=Path, default=Path("gen_results/queries.json"))
parser.add_argument("--ir_hits_path", "-i", type=Path, default=Path("../ir_hits"))
parser.add_argument("--out_dir", "-o", type=Path, default=Path("gen_results"))
parser.add_argument(
    "--load_mode", default="hf", choices=["hf", "json"], help="Load from JSON or HF hub"
)
parser.add_argument(
    "--citation",
    action="store_true",
    help="Response will contain citations (if false, direct answer only)",
)
parser.add_argument("--dry_run", action="store_true", help="Dry run without calling LLMs")
parser.add_argument("--overwrite", action="store_true", help="Overwrite results file if it exists")
parser.add_argument(
    "--relevance_filter",
    choices=[None, "relevant", "non-relevant"],
    default=None,
    help="Filter to relevant or non-relevant documents",
)
parser.add_argument("--skip_no_docs", action="store_true", help="Skip queries with no documents")
parser.add_argument(
    "--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility"
)
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")


def query_openai(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    seed: int = DEFAULT_SEED,
) -> str:
    """
    Sends a prompt to the OpenAI API and returns the generated response.
    Args:
        prompt (str): The input prompt to send to the OpenAI API.
    Returns:
        str: The generated response from the OpenAI API.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    if args.citation:
        # discard old system prompt, use the first part of user_prompt from commandr instead
        system_prompt, user_prompt = user_prompt.split("Query:", 1)
        user_prompt = "Query:" + user_prompt

    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
        seed=seed,
    )

    response = client.chat.completions.create(**kwargs)

    response_str = response.choices[0].message.content.strip()
    return response_str


def generate_next_tokens(input_ids, model, num_tokens=50):
    """
    Generates the next `num_tokens` tokens by repeatedly extending `input_ids`
    and returns log probabilities for all tokens at each step, along with the
    sequence of generated tokens (sampled if temperature > 0).

    Args:
        input_ids (torch.Tensor): The initial input sequence as a tensor.
        model (torch.nn.Module): The language model.
        num_tokens (int): Number of tokens to generate.

    Returns:
        Tuple: (List of numpy arrays, List of generated tokens)
    """
    all_token_probs = []
    generated_tokens = []

    current_input_ids = input_ids

    for step in range(num_tokens):
        with torch.no_grad():
            outputs = model(input_ids=current_input_ids, return_dict=True)

        # Get logits for next token and immediately move to CPU if possible
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  # Shape: [1, vocab_size]

        # Apply temperature to logits
        if args.temperature > 0.0 and args.temperature != 1.0:
            scaled_logits = next_token_logits / args.temperature
        else:
            scaled_logits = next_token_logits

        # Log probabilities for candidate scoring (full vocabulary)
        current_token_log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
        all_token_probs.append(current_token_log_probs.squeeze(0).cpu().numpy())

        # Generate next token
        if args.temperature > 0.0:
            # Sample from the distribution
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding (temperature is 0)
            next_token_id = torch.argmax(scaled_logits, dim=-1, keepdim=True)

        next_token_id_item = next_token_id.item()
        generated_tokens.append(next_token_id_item)

        # Update for next iteration - concatenate efficiently
        current_input_ids = torch.cat([current_input_ids, next_token_id.view(1, 1)], dim=-1)

        # Clear intermediate tensors
        del scaled_logits, current_token_log_probs, next_token_id, outputs, logits

        # Periodic memory cleanup
        if step % 10 == 0:
            torch.cuda.empty_cache()

    return all_token_probs, generated_tokens


def calculate_log_probabilities_for_candidates(all_token_probs, candidates, tokenizer):
    """
    Calculates the log probability for each candidate sequence given a list of
    log probability distributions for each generated token position.

    Args:
        all_token_probs (List of numpy arrays): Log probabilities of all tokens at each step.
        candidates (List of str): List of candidate outputs to evaluate.
        tokenizer (Tokenizer): Tokenizer to convert candidates to token IDs.

    Returns:
        dict: A dictionary where keys are candidate strings and values are their log probabilities.
    """
    candidate_log_probs = []

    for candidate in candidates:
        # Tokenize the candidate sequence to get the list of token IDs
        candidate_ids = tokenizer(
            candidate, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Append the end-of-turn token ID if it exists
        eot_id = tokenizer.convert_tokens_to_ids(
            "<|eot_id|>"
        )  # Get the ID for <|eot_id|> if it exists in the tokenizer
        if eot_id is not None:
            candidate_ids = torch.cat([candidate_ids, torch.tensor([eot_id])])

        # # Reconstruct and print the candidate with the end-of-turn token
        # candidate_reconstructed = tokenizer.decode(candidate_ids, skip_special_tokens=False)
        # print("RECONSTRUCTED WITH EOT:", candidate_reconstructed)

        # Initialize total log probability for the candidate
        total_log_prob = 0.0

        # Calculate log probability for each token in the candidate sequence
        for i, token_id in enumerate(candidate_ids):
            # Ensure we have log probabilities for each position in `all_token_probs`
            if i < len(all_token_probs):
                # Get log probability for the specific token at this position
                total_log_prob += all_token_probs[i][token_id.item()]
            else:
                # If candidate is longer than generated tokens, we cannot calculate further
                print(
                    f"Warning: Candidate '{candidate}' is longer than generated tokens. Increase num_tokens and rerun, previous results are saved."
                )
                exit(1)
                break

        # Store the total log probability for this candidate
        candidate_log_probs.append(total_log_prob)

    return candidate_log_probs


def query_hf(
    user_prompt: str,
    system_prompt: str,
    model_details: ModelDetails,
    claimants: list[str],
    verbose: bool = False,
) -> dict[str, str | int]:
    tokenizer = model_details.tokenizer
    model = model_details.model
    device = model_details.device
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    if args.citation:
        # don't use system prompt for citation
        inputs = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        len_inputs = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": 512,
            "pad_token_id": pad_token_id,
        }

        if args.temperature > 0.0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            output = model.generate(**generation_kwargs)
        raw_response = tokenizer.decode(output.sequences[0][len_inputs:], skip_special_tokens=True)
        return parse_citation_responses(raw_response, claimants)
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        # Get log probabilities and generated tokens using temperature
        token_probs, generated_tokens = generate_next_tokens(input_ids, model, num_tokens=50)

        choices = get_multiple_choice(claimants)

        completion_log_prob = calculate_log_probabilities_for_candidates(
            token_probs, choices, tokenizer
        )

        # Decode the generated token sequence for readability
        generated_sequence = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        # Sort the choices by log probability in descending order
        sorted_log_probs = sorted(
            zip(choices, completion_log_prob), key=lambda x: x[1], reverse=True
        )

        if verbose:
            # Print sorted choices with log probabilities
            print("Sorted by log probability:")
            for choice, log_prob in sorted_log_probs:
                print(f"{choice}: {log_prob}")

            # Print the generated sequence of tokens
            print("\nGenerated sequence:", repr(generated_sequence))

        # Return the most probable claimant based on (potentially tempered) log-probs
        max_index = completion_log_prob.index(max(completion_log_prob))

        # The raw_response is the actual generated sequence, potentially cut at EOS
        if tokenizer.eos_token:
            raw_response = generated_sequence.split(tokenizer.eos_token, 1)[0]
        else:
            raw_response = generated_sequence

        return {
            "choice": choices[max_index],
            "raw_response": raw_response,
            "claimant_ind": max_index,
        }


def parse_citation_responses(raw_response: str, claimants: list[str]) -> dict[str, str | int]:
    max_index = -1
    if not ("Chosen Option:" in raw_response and "Explanation:" in raw_response):
        print("Error: could not parse response below:")
        print(raw_response)
        return {
            "choice": "Unknown",
            "raw_response": raw_response,
            "claimant_ind": -1,
        }

    _, rest = raw_response.split("Chosen Option:", 1)
    llm_choice, explain = rest.split("Explanation:", 1)
    llm_choice, explain = llm_choice.strip(), explain.strip()
    choices = get_multiple_choice(claimants)
    for c in choices:
        if c in llm_choice:
            max_index = choices.index(c)
            break

    if max_index == -1:
        letter_ind, _ = match_one_letter(llm_choice)
        if letter_ind is not None and letter_ind < len(claimants):
            max_index = letter_ind
    if max_index == -1:
        print("WARNING: could not find a match for the response below:")
        print(raw_response)

    return {
        "choice": choices[max_index],
        "raw_response": raw_response,
        "claimant_ind": max_index,
    }


def llama_docs_query_processing(
    docs: List[str], doc_languages, query_entry: dict[str, str | list]
) -> tuple[bool, List[str]]:
    llama_official_languages = ["en", "de", "fr", "it", "hi", "es", "pt", "th"]
    valid_query = True if query_entry["query_lang"] in llama_official_languages else False

    valid_docs = []
    for i in range(len(docs)):
        if doc_languages[i] in llama_official_languages:
            valid_docs.append(docs[i])

    return valid_query, valid_docs


def get_llm_territory(
    query: str,
    docs: List[str],
    doc_languages: List[str],
    query_entry: dict[str, str | list],
    model_details: tuple,
) -> dict[str, str | int]:
    model_name = model_details.model_name
    # TODO: probably need for aya as well
    # if model_name == "llama":
    #     query_valid, docs = llama_docs_query_processing(docs, doc_languages, query_entry)
    #     if not query_valid or len(docs) == 0:
    #         return ""

    use_docs = args.retrieval_over != "no_ir"
    user_prompt = get_user_prompt(
        docs, query_entry, args.citation, use_docs, tokenizer=model_details.tokenizer
    )
    claimants_native = query_entry["claimants_native"]
    system_prompt = SYSTEM_PROMPT if args.role == "vanilla" else SYSTEM_PROMPT_UN_PEACEKEEPER

    response_d = {
        "choice": "Unknown",
        "raw_response": "",
        "claimant_ind": -1,
    }
    # for local LLMs, use rank classification to always get an answer
    if args.dry_run:
        return response_d

    # parse HF models' responses with rank-classification
    if "gpt" not in model_name:
        return query_hf(user_prompt, system_prompt, model_details, claimants_native)

    # otherwise, parse response from OpenAI
    raw_response = query_openai(user_prompt, system_prompt, model_details.model_name, args.seed)

    if args.citation:
        return parse_citation_responses(raw_response, claimants_native)
    else:
        response_d["raw_response"] = raw_response
        answer = raw_response
        letter_ind, _ = match_one_letter(answer)

        if letter_ind is not None and letter_ind < len(claimants_native):
            response_d["choice"] = claimants_native[letter_ind]
            response_d["claimant_ind"] = letter_ind
        elif answer in claimants_native:
            response_d["choice"] = answer
            response_d["claimant_ind"] = claimants_native.index(answer)
        else:
            print("Answer was not found in the query options; setting to Unknown")
            print("Query:", query)
            print("Raw response:", raw_response)
        return response_d


def run_rag(
    data_dict: dict,
    query_dict: dict,
    model_details: tuple,
    out_path: Path,
) -> None:
    """
    Runs RAG for a given query.
    """
    # Load existing results if the file exists, to avoid overwriting
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path, "r") as json_file:
            results = json.load(json_file)
        print(f"Loaded {len(results)} results from {out_path}, continuing")
        num_hits_per_query = [len(results[query_id]["doc_ids"]) for query_id in results]
        num_loaded = len(results)
    else:
        results = {}
        num_hits_per_query = []
        num_loaded = 0

    if data_dict.keys() != query_dict.keys() and args.retrieval_over != "no_ir":
        print(
            f"WARNING:Different number of keys in dicts: query_dict: {len(query_dict)}, data_dict: {len(data_dict)}"
        )
    count_control = 0
    count_one_claimant_lang = 0
    is_ablate_swap = args.retrieval_over == "ablate_swap"

    pbar = tqdm(query_dict.items(), total=len(query_dict))
    for query_id, query_entry in query_dict.items():
        if query_id in results:
            pbar.update(1)
            continue
        pbar.set_description(f"{query_id}")

        query = query_entry["query"]
        query_lang = query_entry["query_lang"]
        claimant_langs = query_entry["claimant_langs"]

        if args.retrieval_over == "no_ir":
            docs = []
            doc_languages = []
            doc_ids = []
        else:
            if is_ablate_swap:
                other_lang = get_other_lang(query_lang, claimant_langs)
                if other_lang is None:
                    # this is a control query, so no other lang, skip
                    count_control += 1
                    pbar.update(1)
                    continue
                if len(claimant_langs) == 1:
                    # all claimants have 1 lang, skip
                    count_one_claimant_lang += 1
                    pbar.update(1)
                    continue

                prefix = query_id.rsplit("_", 1)[0]
                other_query_id = f"{prefix}_{other_lang}"

                dd_entry = data_dict[other_query_id]
            else:
                dd_entry = data_dict[query_id]
            hits = dd_entry["hits"]
            docs = [hit["text"] for hit in hits]
            doc_languages = [hit["language"] for hit in hits]
            doc_ids = [hit["pid"] for hit in hits]
            num_hits_per_query.append(len(docs))
        if args.skip_no_docs and len(docs) == 0:
            pbar.update(1)
            continue

        response_d = get_llm_territory(query, docs, doc_languages, query_entry, model_details)
        # Store the result in the dictionary
        results[query_id] = {
            "query": query,
            "choice": response_d["choice"],
            "raw_response": response_d["raw_response"],
            "claimant_ind": response_d["claimant_ind"],
            "doc_ids": doc_ids,
        }
        if is_ablate_swap:
            results[query_id]["other_lang"] = other_lang

        pbar.update(1)

        # Save results to the JSON file after each query
        with open(out_path, "w") as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
    if args.relevance_filter:
        mean_num_hits = mean(num_hits_per_query)
        stdev_num_hits = stdev(num_hits_per_query)
        print(f"Mean # passages per query: {mean_num_hits:.2f} (stdev: {stdev_num_hits:.2f})")
        if args.skip_no_docs:
            num_hits_nonzero = [n for n in num_hits_per_query if n > 0]
            mean_num_hits_nonzero = mean(num_hits_nonzero)
            stdev_num_hits_nonzero = stdev(num_hits_nonzero)
            print(
                f"Mean # passages per query (non-zero): {mean_num_hits_nonzero:.2f} (stdev: {stdev_num_hits_nonzero:.2f})"
            )
            print(
                f"Ran {len(num_hits_nonzero)}/{len(num_hits_per_query)} queries with non-zero hits"
            )
    if num_loaded != len(results):
        print(f"Results for {len(results)} territories saved to {out_path}")
    pbar.close()
    return count_control, count_one_claimant_lang


def main():
    if args.load_mode == "hf":
        # for ablation, need docs from qlang -- we swap later in code
        retrieval_over = args.retrieval_over if args.retrieval_over != "ablate_swap" else "qlang"
        data_dict = load_bordirlines(
            retrieval_over, args.num_paragraphs, "openai", args.relevance_filter
        )
    elif args.load_mode == "json":
        data_dict = load_json_files(args.ir_hits_path, args.retrieval_over, args.num_paragraphs)

    # Load query metadata
    if not args.query_path.exists():
        print(f"Query metadata file {args.query_path} not found.")
        exit(-1)

    with args.query_path.open("r") as f:
        query_dict = json.load(f)
    query_dict = clean_query_dict(query_dict)

    model_details = get_model_details(args.llm)

    prefix = "generation" if not args.citation else "gencitation"
    seed_name = f"_seed{args.seed}" if args.seed != DEFAULT_SEED else ""
    fname = f"{prefix}_{args.retrieval_over}_{args.role}_{args.llm}_{args.num_paragraphs}{seed_name}.json"

    out_path = args.out_dir / fname
    # Run RAG System
    count_control, count_one_claimant_lang = run_rag(data_dict, query_dict, model_details, out_path)

    if count_control > 0:
        print(f"Skipped {count_control} control queries")
    if count_one_claimant_lang > 0:
        print(f"Skipped {count_one_claimant_lang} queries with only one claimant language")


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"Using {torch.cuda.device_count()} GPUs")
    set_seed(args.seed)
    main()
