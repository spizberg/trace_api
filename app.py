from flask import Flask, request, render_template, redirect, url_for
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import TraceDataset
from utilities import convert_bytes_to_image, convert_image_to_bytes
import torch


app = Flask(__name__)

model, first_time = None, True
mmseg_config_file = "configs/config_model.py"
mmseg_checkpoint_file = "checkpoints/ckpt_model.pth"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/initialize/', methods=['GET'])
def initialize():
    global model, first_time
    if first_time:
        model = init_segmentor(mmseg_config_file, mmseg_checkpoint_file)
        first_time = False
    return redirect("/test/")

@app.route('/test/', methods=['GET', "POST"])
def test():
    if first_time:
        return redirect(url_for('index'))
    if request.method == "POST":
        if not request.files.get('file', ''):
            return redirect(request.url)
        torch.cuda.empty_cache()
        try:
            file = request.files.get('file', '')
            img = convert_bytes_to_image(file.read())
            result = inference_segmentor(model, img)
            output_image = model.show_result(img, result, palette=TraceDataset.PALETTE[:33])
            return render_template('result.html', output_image=convert_image_to_bytes(output_image))
        except:
            return redirect(request.url)
    return render_template('testing.html')


if __name__ == '__main__':
    app.run()
