from flask import Flask, request, render_template
import algo

app = Flask(__name__)


@app.route('/<videoId>', methods=['GET'])
def run_app(videoId):
    data = algo.get_comments_and_topics(videoId)
    return render_template('index.html', dic = data['dic'], topics = data['topics'], len=len(data['dic']))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
