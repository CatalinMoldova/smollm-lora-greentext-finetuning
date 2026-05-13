import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_sources(path):
    notebook = json.loads((ROOT / path).read_text())
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def find_cell(sources, marker):
    matches = [source for source in sources if marker in source]
    if len(matches) != 1:
        raise AssertionError(f"Expected one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


class LoraEvaluationStaticTest(unittest.TestCase):
    def test_python_export_uses_current_training_history(self):
        source = (ROOT / "lora_sft_assignment_final.py").read_text()

        self.assertNotIn("api.run(", source)
        self.assertIn("s16, l16 = loss_curve(trainer)", source)

    def test_python_export_generates_ablation_with_ablation_model(self):
        source = (ROOT / "lora_sft_assignment_final.py").read_text()

        main_section = source.split('print("After FINE-TUNING (base SmolLM-135M + LoRA)")', 1)[1]
        main_section = main_section.split('"""**Remarks**', 1)[0]
        self.assertIn("generate_response(model, tokenizer, p)", main_section)

        ablation_section = source.split('print("After FINE-TUNING Ablation (base SmolLM-135M + LoRA)")', 1)[1]
        ablation_section = ablation_section.split('"""Comparison of perplexity graphs"""', 1)[0]
        self.assertIn("generate_response(model_ab, tokenizer, p)", ablation_section)

    def test_final_notebooks_use_current_training_history(self):
        for notebook in [
            "LoRA_SFT_Assignment_final.ipynb",
            "LoRA_SFT_Assignment_final (4).ipynb",
        ]:
            with self.subTest(notebook=notebook):
                cell = find_cell(notebook_sources(notebook), "s16, l16")
                self.assertNotIn("api.run(", cell)
                self.assertIn("s16, l16 = loss_curve(trainer)", cell)

    def test_final_notebooks_generate_ablation_with_ablation_model(self):
        for notebook in [
            "LoRA_SFT_Assignment_final.ipynb",
            "LoRA_SFT_Assignment_final (4).ipynb",
        ]:
            with self.subTest(notebook=notebook):
                sources = notebook_sources(notebook)
                main_cell = find_cell(sources, "After FINE-TUNING (base SmolLM-135M + LoRA)")
                ablation_cell = find_cell(sources, "After FINE-TUNING Ablation")

                self.assertIn("generate_response(model, tokenizer, p)", main_cell)
                self.assertIn("generate_response(model_ab, tokenizer, p)", ablation_cell)


if __name__ == "__main__":
    unittest.main()
