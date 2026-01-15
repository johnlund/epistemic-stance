#!/usr/bin/env python3
"""
Inference script for the fine-tuned epistemic stance classifier.

Usage:
    # Single text
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final \
        --text "I think everyone should form their own opinion on this matter."
    
    # Batch inference on CSV
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final \
        --input responses.csv --output classified.csv
    
    # Interactive mode
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final --interactive
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System prompt (must match training)
SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework. Analyze the given text and classify it into one of three categories:

**absolutist**: Knowledge presented as certain facts. Claims made without qualification or hedging. Counterarguments dismissed or ignored. High confidence in singular correct answers.

**multiplist**: Knowledge treated as subjective opinion. Multiple perspectives acknowledged but not evaluated against each other. Non-committal stance. "Everyone has their own truth" or "it depends on the person" framing without further analysis.

**evaluativist**: Knowledge as reasoned judgment. Evidence weighed and evaluated. Counterarguments engaged substantively. Qualified language showing awareness of complexity. Conclusions drawn while acknowledging uncertainty.

Respond with a JSON object containing:
- "stance": one of "absolutist", "multiplist", or "evaluativist"
- "confidence": one of "high", "medium", or "low"
"""


class EpistemicStanceClassifier:
    """Wrapper for the fine-tuned epistemic stance classifier."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def classify(self, text: str, max_new_tokens: int = 50) -> dict:
        """Classify a single text and return stance + confidence."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the epistemic stance in the following text:\n\n{text}"}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for classification
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return {
                "stance": result.get("stance", "unknown"),
                "confidence": result.get("confidence", "unknown"),
                "raw_response": response
            }
        except json.JSONDecodeError:
            # Try to extract stance from malformed response
            response_lower = response.lower()
            stance = "unknown"
            for s in ["absolutist", "multiplist", "evaluativist"]:
                if s in response_lower:
                    stance = s
                    break
            
            return {
                "stance": stance,
                "confidence": "low",
                "raw_response": response,
                "parse_error": True
            }
    
    def classify_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> list[dict]:
        """Classify multiple texts."""
        
        results = []
        iterator = tqdm(texts, desc="Classifying") if show_progress else texts
        
        for text in iterator:
            result = self.classify(text)
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run epistemic stance classification')
    parser.add_argument('--model', '-m', required=True, help='Path to fine-tuned model')
    parser.add_argument('--text', '-t', help='Single text to classify')
    parser.add_argument('--input', '-i', help='Input CSV with texts to classify')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--text-column', default='text', help='Column name for text in CSV')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Load model
    classifier = EpistemicStanceClassifier(args.model, device=args.device)
    
    if args.text:
        # Single text classification
        result = classifier.classify(args.text)
        print(f"\nStance: {result['stance']}")
        print(f"Confidence: {result['confidence']}")
        if result.get('parse_error'):
            print(f"Warning: Response parsing failed")
            print(f"Raw response: {result['raw_response']}")
    
    elif args.input:
        # Batch classification
        if not args.output:
            args.output = args.input.replace('.csv', '_classified.csv')
        
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        
        texts = df[args.text_column].tolist()
        results = classifier.classify_batch(texts)
        
        # Add results to dataframe
        df['predicted_stance'] = [r['stance'] for r in results]
        df['prediction_confidence'] = [r['confidence'] for r in results]
        df['parse_error'] = [r.get('parse_error', False) for r in results]
        
        # Save
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("Classification Summary")
        print("=" * 50)
        print(df['predicted_stance'].value_counts())
        print(f"\nParse errors: {df['parse_error'].sum()}")
    
    elif args.interactive:
        # Interactive mode
        print("\nEpistemic Stance Classifier - Interactive Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            text = input("Enter text to classify:\n> ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            result = classifier.classify(text)
            print(f"\n  Stance: {result['stance']}")
            print(f"  Confidence: {result['confidence']}\n")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
