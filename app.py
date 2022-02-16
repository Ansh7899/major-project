from flask import Flask, request, jsonify, render_template, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize',methods=['POST'])
def recognize():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            print(file)

    return render_template('index.html', transcript="Hellow World" , transcript1 = 'Hello world1')
    

if __name__ == "__main__":
    app.run(debug=True, threaded=True)