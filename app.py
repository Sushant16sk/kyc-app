import os
import base64
import numpy as np

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from PIL import Image

import cv2
import easyocr
from flask import send_from_directory
import face_recognition

from models import db, User, Document

# ── App & Config ────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY                    = os.urandom(24).hex(),
    SQLALCHEMY_DATABASE_URI       = 'postgresql://postgres:1234@localhost:5432/kycdb',
    SQLALCHEMY_TRACK_MODIFICATIONS= False,
    UPLOAD_FOLDER                 = os.path.join(app.root_path, 'uploads')
)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

db.init_app(app)
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    db.create_all()

# ── Load Face Detector & OCR ────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ── Helpers ──────────────────────────────────────────────────────
def calculate_ear(eye):
    # Eye Aspect Ratio for blink detection
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('register.html')


@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    age  = int(request.form['age'])

    # 1) Check for webcam snapshot first
    b64 = request.form.get('register_image_data')
    if b64:
        header, encoded = b64.split(',',1)
        img_data = base64.b64decode(encoded)
        filename = f"selfie_{name}_{age}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(img_data)
    else:
        # 2) Fallback to file upload
        imgf = request.files.get('image')
        if not imgf:
            flash('Please upload or capture a face image', 'danger')
            return redirect(url_for('index'))
        filename = secure_filename(imgf.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imgf.save(filepath)

    # → existing face detection + encoding logic follows
    img  = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
    if not len(faces):
        os.remove(filepath)
        flash('No face detected. Try again.', 'danger')
        return redirect(url_for('index'))

    encs = face_recognition.face_encodings(
      face_recognition.load_image_file(filepath)
    )
    if not encs:
        os.remove(filepath)
        flash('Couldn’t encode face. Try a clearer photo.', 'danger')
        return redirect(url_for('index'))

    user = User(
      name          = name,
      age           = age,
      face_image    = filename,
      face_encoding = encs[0].tolist()
    )
    db.session.add(user)
    db.session.commit()

    return redirect(url_for('liveness', user_id=user.id))



@app.route('/liveness/<int:user_id>')
def liveness(user_id):
    return render_template('liveness.html', user_id=user_id)


@app.route('/liveness_upload/<int:user_id>', methods=['POST'])
def liveness_upload(user_id):
    user = User.query.get_or_404(user_id)

    # 1) Read the base64 data from the hidden form field
    b64 = request.form.get('liveness_image_data')
    if not b64:
        flash('No image captured. Try again.', 'danger')
        return redirect(url_for('liveness', user_id=user_id))

    # 2) Decode & save to disk
    header, encoded = b64.split(',', 1)
    img_data = base64.b64decode(encoded)
    filename = f"liveness_{user_id}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(img_data)

    # 3) Face‐to‐registration match
    liv_img  = face_recognition.load_image_file(filepath)
    liv_encs = face_recognition.face_encodings(liv_img)
    if not liv_encs:
        flash('No face found in selfie. Try again.', 'danger')
        return redirect(url_for('liveness', user_id=user_id))

    orig_enc   = np.array(user.face_encoding)
    dist       = face_recognition.face_distance([orig_enc], liv_encs[0])[0]
    face_match = (dist < 0.6)

    # 4) Blink detection (EAR) — with a more forgiving threshold
    landmarks = face_recognition.face_landmarks(liv_img)
    if landmarks:
        lm        = landmarks[0]
        left_ear  = calculate_ear(lm['left_eye'])
        right_ear = calculate_ear(lm['right_eye'])
        ear       = (left_ear + right_ear) / 2.0
        # raise threshold from 0.22 to 0.3
        blink_ok  = (ear < 0.30)
    else:
        blink_ok = False

    # 5) Save liveness result
    user.liveness_image = filename
    user.liveness_pass  = (face_match and blink_ok)

    if user.liveness_pass:
        # replace the original registration encoding with the liveness one
        user.face_encoding = liv_encs[0].tolist()

    db.session.commit()

    if not user.liveness_pass:
        flash('Liveness check failed. Please try again.', 'danger')
        return redirect(url_for('liveness', user_id=user_id))

    flash('Liveness verified!', 'success')
    return redirect(url_for('upload_document', user_id=user_id))

@app.route('/document/<int:user_id>')
def upload_document(user_id):
    user = User.query.get_or_404(user_id)

    # Prefer the liveness selfie if it exists and liveness_pass is True
    selfie_file = user.liveness_image if (user.liveness_pass and user.liveness_image) else user.face_image

    return render_template(
        'document.html',
        user_id=user_id,
        selfie=selfie_file            # ← NEW
    )



@app.route('/document_upload/<int:user_id>', methods=['POST'])
def document_upload(user_id):
    user = User.query.get_or_404(user_id)
    docf = request.files.get('document')
    if not docf:
        flash('Please upload your ID document', 'danger')
        return redirect(url_for('upload_document', user_id=user_id))

    filename = secure_filename(docf.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    docf.save(filepath)

    # OCR via EasyOCR
    results = ocr_reader.readtext(filepath)
    text    = " ".join([res[1] for res in results])

    # Simple name‐in‐text check
    valid = user.name.lower() in text.lower()



# --- new: face-to-ID match ---
    # load and encode the face on the ID
    id_image = face_recognition.load_image_file(filepath)
    id_encs  = face_recognition.face_encodings(id_image)
    if id_encs:
        # compare distance to the registered selfie
        reg_enc = np.array(user.face_encoding)
        dist    = face_recognition.face_distance([reg_enc], id_encs[0])[0]
        face_match = (dist < 0.6)
    else:
        face_match = False
    # Face‐to‐ID match
    id_encs = face_recognition.face_encodings(
        face_recognition.load_image_file(filepath)
    )
    if id_encs:
        dist       = face_recognition.face_distance(
                        [np.array(user.face_encoding)], id_encs[0]
                    )[0]
        face_match = (dist < 0.6)
    else:
        face_match = False

    doc = Document(
        user_id    = user_id,
        filename   = filename,
        ocr_text   = text,
        valid      = valid,
        face_match = face_match
    )
    db.session.add(doc)
    db.session.commit()

    return redirect(url_for('success', user_id=user_id))


@app.route('/success/<int:user_id>')
def success(user_id):
    user = User.query.get_or_404(user_id)
    doc  = user.documents[-1]
    return render_template(
        'success.html',
        liveness_pass = user.liveness_pass,
        doc_valid     = doc.valid,
        face_match    = doc.face_match
    )


if __name__ == '__main__':
    app.run(debug=True)
