def get_model(name):
    m_name="model"
    mod = __import__('{}.{}'.format(__name__, name), fromlist=[''])
    return getattr(mod, m_name)