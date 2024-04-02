META_GENERATE_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""


WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""


ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"""You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is [0.20, 0.39, 0.80, 0.41].
Please perform OCR on the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"output_v7/long_text_OCR/annotated_images/iheartdogs.com_start6864_annotated53.png"


ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_PREDICTION_COT_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should first show your thinking and then tell me your choice in a single uppercase letter. Your answer must consists of two lines, i.e., thinking and answer, and strictly adhere to the following format. Do NOT generate any other contents.
Thinking: <your thinking>
Answer: <your answer>
"""

ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""


ELEMENT_GROUND_BBOX_PROMPT = """"In this UI screenshot, what is the position of the element corresponding to the desctiption: \"{element_desc}\"? 
Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain:
[x1, y1, x2, y2], where ``0 <= x1 < x2 <=1`` and ``0 <= y1 < y2 <=1``. Your output represents the point of the corresponding element in the form of [x1, y1, x2, y2], each value is a [0, 1] decimal number indicating the ratio of the corresponding position to the width or height of the image. 
"""


ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""


ACTION_GROUND_BBOX_PROMPT = """Within the provided UI screenshot, please pinpoint the precise bounding box coordinates of the HTML element's region that I should click in order to complete the following task:
{instruction}
"""

ACTION_GROUND_COT_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should first show your thinking and then tell me your choice in a single uppercase letter. Your answer must consists of two lines, i.e., thinking and answer, and strictly adhere to the following format. Do NOT generate any other contents.
Thinking: <your thinking>
Answer: <your answer>
"""


WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

WEBQA_COT_PROMPT = """{question}
You should first show your thinking and then tell me your answer in the fewest words possible. Your answer must consists of two lines, i.e., thinking and answer, and strictly adhere to the following format. Do NOT generate any other contents.
Thinking: <your thinking>
Answer: <your answer>
"""

ELEMENT_GROUND_POINT_PROMPT = """
In this UI screenshot, what is the position of the element corresponding to the description \"{element_desc}\" (with point)?
"""

ACTION_GROUND_POINT_PROMPT = """
In this UI screenshot, what is the position of the element corresponding to the command \"{instruction}\" (with point)?
"""

DEFAULT_PROMPTS = {
    "meta_generate_prompt": META_GENERATE_PROMPT,
    "web_caption_prompt": WEB_CAPTION_PROMPT,
    "element_ocr_prompt": ELEMENT_OCR_PROMPT,
    "action_prediction_prompt": ACTION_PREDICTION_PROMPT,
    "action_prediction_cot_prompt": ACTION_PREDICTION_COT_PROMPT,
    "element_ground_prompt": ELEMENT_GROUND_PROMPT,
    "action_ground_prompt": ACTION_GROUND_PROMPT,
    "action_ground_cot_prompt": ACTION_GROUND_COT_PROMPT,
    "webqa_prompt": WEBQA_PROMPT,
    "webqa_cot_prompt": WEBQA_COT_PROMPT,
    "element_ground_bbox_prompt": ELEMENT_GROUND_BBOX_PROMPT,
    "action_ground_bbox_prompt": ACTION_GROUND_BBOX_PROMPT,
    "element_ground_point_prompt": ELEMENT_GROUND_POINT_PROMPT,
    "action_ground_point_prompt": ACTION_GROUND_POINT_PROMPT
}