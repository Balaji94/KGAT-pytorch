import logging
import os

import pandas as pd
from flask import Flask, jsonify, render_template
from parser.parser_kgat import *
from kgat_wrapper import KGAT_wrapper
from data_loader.loader_kgat import DataLoaderKGAT

app = Flask("KGAT", template_folder="client")
args = None
kgat_wrapper = None
lookup = pd.read_csv('./datasets/recruitment/lookup_table.csv')
jobs = pd.read_csv('./datasets/recruitment/jobs.csv')
candidates = pd.read_csv('./datasets/recruitment/candidates.csv')
styled_html_table = """
<style>
    th, td {
        padding: 8px;
    }
    th {
        background-color: #f2f2f2;
    }
</style>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>
"""

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
    zipped = zip(ids[1], cf_scores[0])
    sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
    top_k_recommendations = dict(sorted_[:top_k])

    job_org_id = lookup[lookup['id'] == int(jobid)]['org_id'].iloc[0]
    job_details = jobs[jobs['job_id'] == job_org_id]

    top_candidate_ids = top_k_recommendations.keys()
    top_candidates_org_ids = lookup[lookup['id'].isin(top_candidate_ids)]['org_id']
    top_candidates_org_ids = [int(x) for x in top_candidates_org_ids]
    candidates_details = candidates[candidates['Respondent'].isin(top_candidates_org_ids)]
    res = styled_html_table
    res += '<h3> Job Details </h3>'
    res += job_details.to_html()
    res += '<h3> Recommended candidates </h3>'
    res += candidates_details.to_html()
    return res

@app.route("/rj/<candidateid>", methods=['GET'])
def rj(candidateid):
    top_k = 10
    top_k_recommendations = []
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(candidate_id=candidateid)
    zipped = zip(ids[1], cf_scores[0])
    sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
    top_k_recommendations = dict(sorted_[:top_k])
    return str(top_k_recommendations)


@app.route("/check_similarity", methods=['GET'])
def check_similarity():
    return render_template("html/compare_form.html")

@app.route("/compare/<id1>/<r>/<id2>", methods=['GET'])
def compare(id1, r, id2):
    id1_idx = kgat_wrapper.data.users_entities.tolist().index(int(id1))
    id2_idx = kgat_wrapper.data.users_entities.tolist().index(int(id2))

    print(len(kgat_wrapper.data.users_entities_ids))

    similarity = kgat_wrapper.compare(id1_idx, int(r), id2_idx)
    return str(similarity)


if __name__ == '__main__':

    args = parse_kgat_args()
    args.pretrain_model_path = args.pretrain_model_path

    kgat_wrapper = KGAT_wrapper(args)
    kgat_wrapper.data = DataLoaderKGAT(args, logging)

    app.run(port=8000, debug=False)
