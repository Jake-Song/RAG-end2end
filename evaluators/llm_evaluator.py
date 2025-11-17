"""
LLM as Judge 방식
1. correctness 평가
2. relevance 평가
3. groundedness 평가
4. retrieval_relevance 평가
"""

from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

class CorrectnessEvaluator:
    def __init__(self):
        # self.grader_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
        #     CorrectnessGrade, method="json_schema", strict=True
        # )

        self.grader_llm = ChatUpstage(model="solar-pro2", temperature=0.0).with_structured_output(
            CorrectnessGrade
        )
        self.correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
        (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
        (3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.
        Correctness:
        A correctness value of True means that the student's answer meets all of the criteria.
        A correctness value of False means that the student's answer does not meet all of the criteria.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    def correctness(self, inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        print("inputs", inputs)
        print("reference_outputs" + "\n", reference_outputs)
        print("outputs" + "\n", outputs)
        """An evaluator for RAG answer accuracy"""
        answers = f"""\
        QUESTION: {inputs['query']}
        GROUND TRUTH ANSWER: {reference_outputs['answer']}
        STUDENT ANSWER: {outputs['answer']}"""
        # Run evaluator
        grade = self.grader_llm.invoke([
            {"role": "system", "content": self.correctness_instructions},
            {"role": "user", "content": answers}
        ])
        return {"correctness": grade["correct"], "explanation": grade["explanation"]}

class RelevanceEvaluator:
    def __init__(self):
        self.grader_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
            RelevanceGrade, method="json_schema", strict=True
        )
        self.relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
        (2) Ensure the STUDENT ANSWER helps to answer the QUESTION
        Relevance:
        A relevance value of True means that the student's answer meets all of the criteria.
        A relevance value of False means that the student's answer does not meet all of the criteria.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    def relevance(self, inputs: dict, outputs: dict) -> bool:
        """A simple evaluator for RAG answer helpfulness."""
        answer = f"QUESTION: {inputs['query']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = self.grader_llm.invoke([
            {"role": "system", "content": self.relevance_instructions},
            {"role": "user", "content": answer}
        ])
        return grade["relevant"]

class GroundedEvaluator:
    def __init__(self):
        self.grader_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
            GroundedGrade, method="json_schema", strict=True
        )
        self.grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
        Grounded:
        A grounded value of True means that the student's answer meets all of the criteria.
        A grounded value of False means that the student's answer does not meet all of the criteria.
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
        
    def groundedness(self, inputs: dict, outputs: dict) -> bool:
        """A simple evaluator for RAG answer groundedness."""
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = self.grader_llm.invoke([
            {"role": "system", "content": self.grounded_instructions},
            {"role": "user", "content": answer}
        ])
        return grade["grounded"]

class RetrievalRelevanceEvaluator:
    def __init__(self):
        self.grader_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
            RetrievalRelevanceGrade, method="json_schema", strict=True
        )
        self.retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
        (1) You goal is to identify FACTS that are completely unrelated to the QUESTION
        (2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
        (3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met
        Relevance:
        A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
        A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

    def retrieval_relevance(self, inputs: dict, outputs: dict) -> bool:
        """An evaluator for document relevance."""
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        grade = self.grader_llm.invoke([
            {"role": "system", "content": self.retrieval_relevance_instructions},
            {"role": "user", "content": answer}
        ])
        return grade["relevant"]