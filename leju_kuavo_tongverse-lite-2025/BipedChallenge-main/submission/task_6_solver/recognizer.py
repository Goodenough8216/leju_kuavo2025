from openai import OpenAI
import base64
import numpy as np

class Recognizer():
    def __init__(self, api_key) -> None:
        self.client = OpenAI(api_key=api_key, base_url="https://api.stepfun.com/v1")

    """
    查看位置：输入所处房间、图像和要识别的物体，返回物体的位置信息(left/middle/right : string)
    """
    def look(self, room_objects: list ,target: str, img_addr: str) -> str:
        img_str = ""
        with open(img_addr, 'rb') as f:
            img_data = f.read()
            img_str = base64.b64encode(img_data).decode('ascii')
        completion = self.client.chat.completions.create(
          model="step-1v-32k",
          messages=[
              {
                  "role": "system",
                  "content": "你是一个物体位置识别专家，根据用户输入的图片和要识别的物体给出物体的位置信息，位置信息应从下面选择一个：(left, middle, right)，注意不要输出多余信息",
              },
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": "现在图片中是一个桌面，上面有{},{},{}，输出{}相对于这三个物体的位置，你只能输出left, middle, right其中之一，注意不要输出多余信息".format(*room_objects, target),
                      },
                      {
                          "type": "image_url",
                          "image_url": {
                              "url": "data:image/webp;base64,{}".format(img_str),
                          },
                      },
                  ],
              },
          ],
        )
        return completion.choices[0].message.content

