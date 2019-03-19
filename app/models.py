from mongokit import ValidationError, Document


def max_length(length):
    def validate(value):
        if len(value) <= length:
            return True
        # must have %s in error format string to have mongokit place key in there
        raise ValidationError('%s must be at most {} characters long'.format(length))
    return validate


class Hashtags(Document):
    structure = {
        'hashtag': unicode,
        'createdAt': unicode,
        'updatedAt': unicode,
    }
    validators = {
    }
    use_dot_notation = True

    def __repr__(self):
        return '<User %r>' % self.name
