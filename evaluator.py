    
    # # evaluation
    # def parser_function(output_str: str) -> Tuple[float, str]:
    #     # Print result to backtrack
    #     display(Markdown(f"<b>{output_str}</b>"))

    #     # Pattern to match the feedback and response
    #     # This pattern looks for any text ending with '[RESULT]' followed by a number
    #     pattern = r"(.+?) \[RESULT\] (\d)"

    #     # Using regex to find all matches
    #     matches = re.findall(pattern, output_str)

    #     # Check if any match is found
    #     if matches:
    #         # Assuming there's only one match in the text, extract feedback and response
    #         feedback, score = matches[0]
    #         score = float(score.strip()) if score is not None else score
    #         return score, feedback.strip()
    #     else:
    #         return None, None
    
import prompts
import utils
import warnings
from typing import Any, Dict, List, Tuple, Union
import asyncio
        
class PrometheusEval:
    def __init__(
        self,
        model,
        grading_format: str,  # 기본값 없이 필수 인자로 설정
        include_reference: bool,
        include_context: bool
    ):
        self.is_async = False  # 모델이 비동기적인지 여부를 나타내는 플래그
        self.model = model  # 평가에 사용될 모델

        # grading_format 검증
        if grading_format not in ["absolute", "relative"]:
            raise ValueError("Invalid grading format. Choose 'absolute' or 'relative'.")

        self.grading_format = grading_format

        # 적절한 템플릿 선택
        self.grade_template = prompts.get_prompt_template(grading_format, include_context, include_reference)

        # 템플릿에 참조 답안 검증
        if grading_format == "absolute" and "###Reference Answer (Score 5):" not in self.grade_template:
            warnings.warn(
                "Reference answer was not given in Absolute Grading mode. This might lead to non-optimal performances."
            )
        if grading_format == "relative" and "###Reference Answer:" not in self.grade_template:
            warnings.warn(
                "Reference answer was not given in Relative Grading mode. This might lead to non-optimal performances."
            )               

        self.absolute_grade_template = prompts.get_prompt_template("absolute", include_context, include_reference)
        self.relative_grade_template = prompts.get_prompt_template("relative", include_context, include_reference)
                
    def single_absolute_grade(
        self,
        instruction: str,
        response: str,
        rubric: str,
        reference_answer: str = None,
        context: str = None,
        params: Dict[str, Any] = {},
    ) -> Tuple[str, int]:
        """
        Grades a single response absolutely based on the provided instruction and response.

        :param instruction: The instruction for the response.
        :param response: The response to grade.
        :param rubric: The rubric to use for grading.
        :param reference_answer: Optional reference answer to aid in grading.
        :param context:
        :param params: Additional parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing the feedback and score.
        """
        
        feedbacks, scores = self.absolute_grade(
            instructions=[instruction],
            responses=[response],
            rubric=[rubric], 
            reference_answers=[reference_answer] if reference_answer else [None],
            contexts=[context] if context else [None],
            params=params,
        )
        return feedbacks[0], scores[0]

    def single_relative_grade(
        self,
        instruction: str,
        response_A: str,
        response_B: str,
        rubric: str,
        reference_answer: str = None,
        params: Dict[str, Any] = {},
    ) -> Tuple[str, int]:
        """
        Grades two responses relatively based on a single instruction.

        :param instruction: The instruction for the paired responses.
        :param response_A: First response to compare.
        :param response_B: Second response to compare.
        :param rubric: The rubric to use for grading.
        :param reference_answers: Optional tuple of reference answers for each response.
        :param params: Additional parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing the feedbacks and scores.
        """
        

        feedbacks, scores = self.relative_grade(
            instructions=[instruction],
            responses_A=[response_A],
            responses_B=[response_B],
            rubric=[rubric],
            reference_answers=[reference_answer] if reference_answer else [None],
            params=params,
        )
        return feedbacks[0], scores[0]

    def _check_inputs(self, instructions, responses, rubric, reference_answers, contexts):
        if len(instructions) != len(responses):
            raise ValueError(
                "Length of instructions must match the length of responses"
            )

        # If rubric is a list, check its length matches the length of instructions
        if isinstance(rubric, list) and len(rubric) != len(instructions):
            raise ValueError("Length of rubric must match the length of instructions")
        elif isinstance(rubric, list) and len(rubric) == len(instructions):
            pass
        elif isinstance(rubric, str):
            rubric = [rubric] * len(
                instructions
            )  # Apply the same rubric to all if it's not a list
        else:
            raise ValueError("Rubric must be a string or a list of strings")

        # Handle reference answers
        if isinstance(reference_answers, list) and len(reference_answers) != len(
            instructions
        ):
            raise ValueError(
                "Length of reference answers must match the length of instructions"
            )
        elif isinstance(reference_answers, list):
            pass
        else:
            reference_answers = [None] * len(
                instructions
            )  # Default to None if not provided

        # Handle contexts
        if isinstance(contexts, list) and len(contexts) != len(
            instructions
        ):
            raise ValueError(
                "Length of reference answers must match the length of instructions"
            )
        elif isinstance(contexts, list):
            pass
        else:
            contexts = [None] * len(
                instructions
            )  # Default to None if not provided
            
        return instructions, responses, rubric, reference_answers, contexts

    def absolute_grade(
        self,
        *,
        instructions: List[str],
        responses: List[str],
        rubric: List[str] | str,
        reference_answers: List[str] = None,
        contexts: List[str] = None,
        params: Dict[str, Any] = {},
    ) -> Tuple[List[str], List[int]]:
        """
        Grades a batch of responses absolutely based on the provided instructions and responses.

        :param instructions: List of instructions corresponding to each response.
        :param responses: List of responses to grade.
        :param params: Parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing lists of feedbacks and scores.
        """

        instructions, responses, rubric, reference_answers, contexts = self._check_inputs(
            instructions, responses, rubric, reference_answers, contexts
        )

        inputs = []
        for idx, (instruction, response) in enumerate(zip(instructions, responses)):
            rubric_ = rubric[idx]
            reference_answer = reference_answers[idx]
            context = contexts[idx]
            content = self.absolute_grade_template.format(
                instruction=instruction,
                response=response,
                rubric=rubric_,
                reference_answer=reference_answer,
                context=context
            )
            messages = [
                {"role": "system", "content": prompts.ABS_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            if hasattr(self.model, "validate_vllm"):
                input_ = self._get_conversation_prompt(messages)
            else:
                input_ = messages
            inputs.append(input_)

        if self.is_async:
            feedbacks, scores = asyncio.run(
                utils.async_batch_completions_with_retries(
                    self.model,
                    inputs,
                    mode="absolute",
                    params=params,
                )
            )
        else:
            feedbacks, scores = utils.batch_completions_with_retries(
                self.model,
                inputs,
                mode="absolute",
                params=params,
            )

        return feedbacks, scores
    
    def relative_grade(
        self,
        *,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        rubric: List[str] | str,
        reference_answers: List[str] = None,
        params: Dict[str, Any] = {},
    ) -> Tuple[List[str], List[int]]:
        """
        Grades a batch of responses relatively based on the provided instructions and paired responses.

        :param instructions: List of instructions for each paired responses.
        :param responses_A: List of first responses in each pair.
        :param responses_B: List of second responses in each pair.
        :param params: Additional parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing lists of feedbacks and scores.
        """

        instructions, _, rubric, reference_answers = self._check_inputs(
            instructions, list(zip(responses_A, responses_B)), rubric, reference_answers
        )

        inputs = []
        for idx, (instruction, response_a, response_b) in enumerate(
            zip(instructions, responses_A, responses_B)
        ):
            rubric_ = rubric[idx]
            reference_answer = reference_answers[idx]
            content = self.relative_grade_template.format(
                instruction=instruction,
                response_A=response_a,
                response_B=response_b,
                rubric=rubric_,
                reference_answer=reference_answer,
            )
            messages = [
                {"role": "system", "content": prompts.REL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            if hasattr(self.model, "validate_vllm"):
                input_ = self._get_conversation_prompt(messages)
            else:
                input_ = messages
            inputs.append(input_)

        if self.is_async:
            feedbacks, scores = asyncio.run(
                utils.async_batch_completions_with_retries(
                    self.model,
                    inputs,
                    mode="relative",
                    params=params,
                )
            )
        else:
            feedbacks, scores = utils.batch_completions_with_retries(
                self.model,
                inputs,
                mode="relative",
                params=params,
            )

        return feedbacks, scores