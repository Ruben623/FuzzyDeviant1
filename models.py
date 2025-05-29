from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    deviance_level = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f'<Result {self.id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'age': self.age,
            'gender': self.gender,
            'deviance_level': self.deviance_level,
            'date': self.date.isoformat()
        }
    
    