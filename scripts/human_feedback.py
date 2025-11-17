"""
Human feedback evaluation CLI for eval results
Shows one row at a time and collects yes/no feedback
"""

import pandas as pd
from config import output_path_prefix
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError


class YesNoValidator(Validator):
    def validate(self, document):
        text = document.text.lower().strip()
        if text not in ['y', 'n', 'yes', 'no']:
            raise ValidationError(
                message='Please enter y/n or yes/no',
                cursor_position=len(document.text)
            )


def display_row(row, index, total):
    """Display a single row's information"""
    print("\n" + "=" * 80)
    print(f"Row {index + 1} / {total}")
    print("=" * 80)
    print(f"\nQuery: {row['query']}")
    print(f"\nGround Truth Answer: {row['answer']}")
    print(f"\nGenerated Answer: {row['outputs.answer']}")
    print(f"\nPage Number: {row['outputs.page_number']}")
    print(f"\nLLM Correctness: {row['correctness']}")
    print(f"\nLLM Explanation: {row['explanation']}")
    print("-" * 80)


def get_feedback():
    """Get yes/no feedback from user"""
    validator = YesNoValidator()
    response = prompt(
        "Is this answer correct? (y/n): ",
        validator=validator
    ).lower().strip()

    return response in ['y', 'yes']


def main():
    # Load evaluation CSV
    eval_file = f"{output_path_prefix}_eval.csv"
    print(f"Loading evaluation file: {eval_file}")

    df = pd.read_csv(eval_file)
    print(f"Loaded {len(df)} rows")

    # Add feedback column if it doesn't exist
    if 'human_feedback' not in df.columns:
        df['human_feedback'] = None

    # Iterate through rows
    start_index = 0
    # Find first row without feedback (if resuming)
    if df['human_feedback'].notna().any():
        start_index = df['human_feedback'].isna().idxmax()
        print(f"Resuming from row {start_index + 1}")

    for i in range(start_index, len(df)):
        row = df.iloc[i]

        # Display row information
        display_row(row, i, len(df))

        try:
            # Get feedback
            feedback = get_feedback()
            df.loc[i, 'human_feedback'] = feedback

            # Save after each feedback
            df.to_csv(eval_file, index=False)
            print(f"✓ Feedback saved: {'Correct' if feedback else 'Incorrect'}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress has been saved.")
            break

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    completed = df['human_feedback'].notna().sum()
    print(f"Total rows evaluated: {completed} / {len(df)}")

    if completed > 0:
        correct = df['human_feedback'].sum()
        print(f"Human marked correct: {correct} / {completed} ({correct/completed*100:.1f}%)")

        # Compare with LLM evaluation
        llm_correct = df[df['human_feedback'].notna()]['correctness'].sum()
        print(f"LLM marked correct: {llm_correct} / {completed} ({llm_correct/completed*100:.1f}%)")

        # Agreement rate
        agreement = (df['human_feedback'] == df['correctness']).sum()
        print(f"Human-LLM agreement: {agreement} / {completed} ({agreement/completed*100:.1f}%)")


if __name__ == "__main__":
    main()
