from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os

import cv2

from main import run_tracker_in_thread
from main import model1

app = Flask(__name__)

app.config['SECRET_KEY'] = 'safezone'
app.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames_web1(path_x):
    yolo_output1 = run_tracker_in_thread(path_x, model1, 1)
    for detection_ in yolo_output1:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_web2(path_x):
    yolo_output2 = run_tracker_in_thread(path_x, model1, 2)
    for detection_ in yolo_output2:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)


# Webcam Output
@app.route('/webapp')
def webapp():
    return Response(generate_frames_web1(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webapp2')
def webapp2():
    return Response(generate_frames_web2(path_x=1), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
