from config.config import cfg
from llm import Language_model
import torch
import os






def main():
    args = cfg()

    model_dir = os.path.join(args.model_base_dir, args.model_instance_dir)
    
    model = Language_model(args, args.model_name, model_dir, args.tokenizer_path, "cuda")

    mafuyu_model = model.prepare_models(quantization_type = "nf4",precision = torch.float16)

    mafuyu_tokenizer = model.prepare_tokenizer()

    test_prompt = model.prepare_prompt(prompt = args.prompt)

    final_prompt = f"""指示:{test_prompt}
    応答:
    """


    input_ids = mafuyu_tokenizer.encode(final_prompt, add_special_tokens=False, return_tensors="pt")

    output_ids = mafuyu_model.generate(
        input_ids=input_ids.to(device=model.device),
        max_length=200,
        temperature=0.7,
        do_sample=True,
    )

    output = mafuyu_tokenizer.decode(output_ids.tolist()[0][input_ids.size(1):])

    print(args.prompt)
    print(output)


if __name__ == "__main__":
    main()