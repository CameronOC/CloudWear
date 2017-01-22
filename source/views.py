# project/main/views.py


#################
#### imports ####
#################
from flask import request, redirect, url_for
from _curses import flash
import os
from werkzeug.utils import secure_filename
from flask import render_template, Blueprint
from flask_login import login_required
from source import app
from flask import send_from_directory
import cStringIO
from werkzeug.datastructures import FileStorage
################
#### config ####
################

main_blueprint = Blueprint('main', __name__,)


################
#### routes ####
################


@main_blueprint.route('/')
@login_required
def home():
    return render_template('index.html')

ALLOWED_EXTENSIONS = set(['jpg,jpeg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_blueprint.route('/upload', methods=['POST'])
def upload_file():
    image_str = request.get_data()
    new_image = image_str.split(',')
    print (image_str)
    decoded_data = new_image[1].decode('base64')
    print (decoded_data)
    file_data = cStringIO.StringIO(decoded_data)
    print(file_data)
    file = FileStorage(file_data, filename='screenshot.jpeg')
    print(file)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER'], filename))
        os.system("");
        os.system("");
    return render_template('index.html')

@main_blueprint.route('/download', methods=['GET'])
def download():
    return app.send_static_file(os.path.join(os.getcwd(),'/clothe_recognition/out_images.json'))

    # if request.method == 'POST':
    #     # check if the post request has the file part
    #     if 'file' not in request.files:
    #         flash('No file part')
    #         return redirect(request.url)
    #     file = request.files['file']
    #     # if user does not select file, browser also
    #     # submit a empty part without filename
    #     if file.filename == '':
    #         flash('No selected file')
    #         return redirect(request.url)
    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #         return redirect(url_for('uploaded_file',
    #                                 filename=filename))