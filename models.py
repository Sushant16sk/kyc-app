from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__      = 'users'
    id                 = db.Column(db.Integer, primary_key=True)
    name               = db.Column(db.String(128), nullable=False)
    age                = db.Column(db.Integer, nullable=False)
    face_image         = db.Column(db.String(256), nullable=False)
    face_encoding      = db.Column(db.PickleType, nullable=False)  # store 128-d vector
    liveness_image     = db.Column(db.String(256), nullable=True)
    liveness_pass      = db.Column(db.Boolean, default=False)

class Document(db.Model):
    __tablename__   = 'documents'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename        = db.Column(db.String(256), nullable=False)
    ocr_text        = db.Column(db.Text, nullable=False)
    valid           = db.Column(db.Boolean, default=False)
    face_match      = db.Column(db.Boolean, default=False)
    user            = db.relationship('User', backref=db.backref('documents', lazy=True))
