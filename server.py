import logging
from flask import Flask, jsonify
from parser.parser_kgat import *
from kgat_wrapper import KGAT_wrapper
from data_loader.loader_kgat import DataLoaderKGAT

app = Flask("KGAT")
args = None
kgat_wrapper = None

@app.route("/")
def home():
    return "<h1>KGAT POC<h1>"

@app.route("/train", methods=['GET'])
def train():
    kgat_wrapper.train()
    return "Training Done!!!"

@app.route("/predict", methods=['GET'])
def predict():
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(args)
    return str(cf_scores) + "\n\n\n" + str(ids)

@app.route("/jobs", methods=['GET'])
def get_jobs():
    return str(kgat_wrapper.data.users)

@app.route("/candidates", methods=['GET'])
def get_candidates():
    return str(kgat_wrapper.data.items)

@app.route("/rc/<jobid>", methods=['GET'])
def rc(jobid):
    top_k = 10
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(job_id=jobid)
    return top_k_recommendations

@app.route("/rj/<candidateid>", methods=['GET'])
def rj(candidateid):
    top_k = 10
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(candidate_id=candidateid)
    return top_k_recommendations



if __name__ == '__main__':

    args = parse_kgat_args()
    args.pretrain_model_path = args.pretrain_model_path.replace("model.pth", "kgat_model_recruit.pth")

    kgat_wrapper = KGAT_wrapper(args)
    kgat_wrapper.data = DataLoaderKGAT(args, logging)

    app.run(port=8000, debug=True)
