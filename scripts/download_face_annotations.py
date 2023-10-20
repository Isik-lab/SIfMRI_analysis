import labelbox as lb
import json

LB_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGl0M3Q5enUwZWVmMDd5bmNndms2NWdhIiwib3JnYW5pemF0aW9uSWQiOiJjbGl0M3Q5emUwZWVlMDd5bjN0YTc0bWR2IiwiYXBpS2V5SWQiOiJjbGl0NWhucTgwNGhhMDd4eWZibjY3Z3JmIiwic2VjcmV0IjoiYzBjMmYwYjcwODEyNTYwYWM5MzgwZmVkZDg2NjNiODkiLCJpYXQiOjE2ODY1OTIyNjksImV4cCI6MjMxNzc0NDI2OX0.NDCbqPXct-Pn4pcqpVIG84PK9Q6Y7u6yTp6-pfBfXQA'
PROJECT_ID = 'clit3zloh00k4071d6x1lc5ej'
client = lb.Client(api_key=LB_API_KEY)
project = client.get_project(PROJECT_ID)
export_params = {
    "data_row_details": True,
    "metadata": False,
    "attachments": False,
    "project_details": False,
    "performance_details": False,
    "label_details": True,
    "interpolated_frames": True
}

export_task = project.export_v2(params=export_params)
export_task.wait_till_done()
if export_task.errors:
  print(export_task.errors)
export_json = export_task.result
print("results: ", export_json)

# Serializing json
with open("../data/raw/face_annotation/annotations.json", "w") as outfile:
    json.dump(export_json, outfile, indent=4)
outfile.close()