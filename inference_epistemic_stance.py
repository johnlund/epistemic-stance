#!/usr/bin/env python3
"""
Inference script for the fine-tuned Mistral-Small-24B-Instruct-2501 epistemic stance classifier.

Usage:
    # Single text classification
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final --text "Your text here"
    
    # Interactive mode
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final --interactive
    
    # Batch classification from CSV
    python inference_epistemic_stance.py --model ./epistemic_stance_model/final --input data.csv --output results.csv
"""

import argparse
import json
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an expert classifier trained to identify epistemic stances in text based on Kuhn's developmental epistemology framework.

## The Three Epistemic Stances

**ABSOLUTIST**: Knowledge is CERTAIN. Claims presented as objective truth without qualification. Dismisses opposing views. Uses directive language like "You need to...", "Obviously...", "The only option is..."

**MULTIPLIST**: Knowledge is SUBJECTIVE. All opinions equally valid. Avoids evaluation. Uses phrases like "Only you can know", "Everyone's entitled to their view", "It depends on the person"

**EVALUATIVIST**: Knowledge is UNCERTAIN but some claims have MORE MERIT. Weighs evidence, engages counterarguments, shows calibrated confidence. Uses "The evidence suggests...", "On balance...", "Based on what you've described..."

## Key Distinctions
- Absolutist vs Evaluativist: Do they JUSTIFY with reasoning, or assert as fact?
- Multiplist vs Evaluativist: Do they WEIGH perspectives, or treat all as equal?

Respond with JSON: {"stance": "absolutist|multiplist|evaluativist", "confidence": "high|medium|low"}"""


class EpistemicStanceClassifier:
    """Classifier for epistemic stances using fine-tuned Mistral Small 3."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if self.device == 'cuda' else None,
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def classify(
        self,
        text: str,
        temperature: float = 0.15,
        max_new_tokens: int = 50,
    ) -> dict:
        """
        Classify the epistemic stance of a text.
        
        Args:
            text: The text to classify
            temperature: Sampling temperature (low recommended for Mistral Small 3)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            dict with 'stance', 'confidence', and 'raw_response'
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the epistemic stance:\n\n{text}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse the response
        try:
            result = json.loads(response)
            return {
                "stance": result.get("stance", "unknown"),
                "confidence": result.get("confidence", "unknown"),
                "raw_response": response,
            }
        except json.JSONDecodeError:
            # Fallback: try to extract stance from text
            stance = "unknown"
            for label in ["absolutist", "multiplist", "evaluativist"]:
                if label in response.lower():
                    stance = label
                    break
            return {
                "stance": stance,
                "confidence": "low",
                "raw_response": response,
            }


def interactive_mode(classifier: EpistemicStanceClassifier):
    """Run interactive classification mode."""
    print("\n" + "="*60)
    print("Epistemic Stance Classifier - Interactive Mode")
    print("="*60)
    print("\nEnter text to classify (or 'quit' to exit):")
    print("-"*60)
    
    while True:
        try:
            print("\n> ", end="")
            text = input().strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text to classify.")
                continue
            
            result = classifier.classify(text)
            
            print(f"\nResult:")
            print(f"  Stance: {result['stance'].upper()}")
            print(f"  Confidence: {result['confidence']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_classify(
    classifier: EpistemicStanceClassifier,
    input_path: str,
    output_path: str,
    text_column: str = "text",
):
    """Classify texts from a CSV file."""
    import pandas as pd
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV")
    
    results = []
    total = len(df)
    
    print(f"Classifying {total} texts...")
    for idx, row in df.iterrows():
        text = row[text_column]
        result = classifier.classify(text)
        results.append({
            "text": text,
            "predicted_stance": result["stance"],
            "confidence": result["confidence"],
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total}")
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Add original columns
    for col in df.columns:
        if col != text_column:
            output_df[col] = df[col].values
    
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\nClassification Summary:")
    print(output_df['predicted_stance'].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Epistemic Stance Classifier")
    parser.add_argument('--model', '-m', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--text', '-t', help='Single text to classify')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--input', help='Input CSV file for batch classification')
    parser.add_argument('--output', help='Output CSV file for batch results')
    parser.add_argument('--text-column', default='text', help='Column name for text in CSV')
    parser.add_argument('--temperature', type=float, default=0.15, help='Sampling temperature')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.text, args.interactive, args.input]):
        parser.error("Must specify --text, --interactive, or --input")
    
    if args.input and not args.output:
        parser.error("--output required when using --input")
    
    # Initialize classifier
    classifier = EpistemicStanceClassifier(args.model, device=args.device)
    
    # Run appropriate mode
    if args.text:
        result = classifier.classify(args.text, temperature=args.temperature)
        print(f"\nText: {args.text[:100]}...")
        print(f"\nResult:")
        print(f"  Stance: {result['stance'].upper()}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Raw response: {result['raw_response']}")
    
    elif args.interactive:
        interactive_mode(classifier)
    
    elif args.input:
        batch_classify(
            classifier,
            args.input,
            args.output,
            text_column=args.text_column,
        )


if __name__ == '__main__':
    main()
