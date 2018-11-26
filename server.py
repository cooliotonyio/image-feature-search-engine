import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './queries'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/queries/<path:filename>')
def queries(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("woot")
            print(redirect(url_for('upload_file',
                                    filename=filename)))
            print("woota")
            return redirect(url_for('upload_file',
                                    filename=filename))
    elif 'filename' in request.args:
        filename = request.args['filename']
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        html = '''
            <!doctype html>
            <title>Results</title>
            <h1>Results</h1>
            <img src="{}"></img>
            '''.format(full_filename)
        return html
    else:
        html = '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
            '''
        return html

if __name__ == '__main__':
    app.debug = True
    app.run()