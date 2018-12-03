import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from search import SearchEngine
from torchvision import transforms
from datasets import ImageFolder
import torch

THRESHOLD = 1
SAVE_DIRECTORY = './binary_embeddings'
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
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], request.args['filename'])
        distances, paths = search_engine.query(full_filename)

        html_imgs = [f"<img src={path}" for path in paths]
        html = '''
            <!doctype html>
            <title>Results</title>
            <h1>Query</h1>
            <img src="{}"></img>
            <h1>Results</h1>
            {}
            '''.format(full_filename, " ".join(html_imgs))
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
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    data = ImageFolder('./Flickr', transform=transform)

    cuda = torch.cuda.is_available()
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)

    search_engine = SearchEngine(data = data, cuda = cuda, threshold = THRESHOLD, save_directory = SAVE_DIRECTORY, transform=transform)
    search_engine.fit(data_loader = data_loader, load_embeddings = True, verbose = True)

    app.debug = False
    app.run(
        host=os.getenv('LISTEN', '0.0.0.0'),
        port=int(os.getenv('PORT', '80'))
    )
