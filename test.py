import argparse
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--prompt", type=str, default="Write me an essay which includes a synopsis of your 5 favorite Shakespeare plays.", help="Prompt")

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")

    tmp_logs = os.path.expanduser("~/vllm/tmp.txt")
    if os.path.exists(tmp_logs): os.remove(tmp_logs)
    
    swa_layers = list(range(100))
    llm = LLM(model=args.model, max_model_len=8192, enforce_eager=True, enable_prefix_caching=False, tensor_parallel_size=1, swa_layers=swa_layers)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply chat template to the prompt
    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)

    outputs = llm.generate([formatted_prompt], sampling_params)
    outputs = list(map(lambda x: x.outputs[0].text, outputs))

    print("\n".join(outputs))


if __name__ == "__main__":
    main()
