import logging
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request
from parser.parser_kgat import *
from kgat_wrapper import KGAT_wrapper
from data_loader.loader_kgat import DataLoaderKGAT

app = Flask("KGAT", template_folder="client")
args = None
kgat_wrapper = None
lookup = pd.read_csv('./datasets/recruitment/lookup_table.csv')
jobs = pd.read_csv('./datasets/recruitment/jobs.csv')
candidates = pd.read_csv('./datasets/recruitment/candidates.csv')

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

@app.route("/rc", methods=['GET','POST'])
def recommend_candidates():
    job_ids = kgat_wrapper.data.users
    return render_template('html/candidate_recommendation.html', job_ids=job_ids)

@app.route('/get_user_details', methods=['GET'])
def get_user_details():
    user_id = int(request.args.get('user_id'))
    job_org_id = lookup[lookup['id'] == int(user_id)]['org_id'].iloc[0]
    job_details = jobs[jobs['job_id'] == job_org_id].drop(columns=['job_id'])
    user_details_html = job_details.to_html()
    return jsonify(user_details_html)

@app.route('/get_item_recommendations', methods=['GET'])
def get_item_recommendations():
    jobid = int(request.args.get('user_id'))
    top_k = 10
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(job_id=jobid)
    zipped = zip(ids[1], cf_scores[0])
    sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
    top_k_recommendations = dict(sorted_[:top_k])
    top_candidate_ids = top_k_recommendations.keys()
    top_candidates_org_ids = lookup[lookup['id'].isin(top_candidate_ids)]['org_id']
    top_candidates_org_ids = [int(x) for x in top_candidates_org_ids]
    candidates_details = candidates[candidates['Respondent'].isin(top_candidates_org_ids)].drop(columns=['Respondent', 'LastJob1', 'LastJob2', 'LastJob3'])
    # candidates_details['score'] = top_k_recommendations.values()
    candidates_details_html = candidates_details.to_html()
    return jsonify(candidates_details_html)

@app.route("/rj", methods=['GET','POST'])
def recommend_jobs():
    candidate_ids = kgat_wrapper.data.items
    return render_template('html/job_recommendation.html', candidate_ids=candidate_ids)

@app.route('/get_item_details', methods=['GET'])
def get_item_details():
    item_id = int(request.args.get('item_id'))
    candidate_org_id = lookup[lookup['id'] == item_id]['org_id'].iloc[0]
    candidate_details = candidates[candidates['Respondent'] == int(candidate_org_id)].drop(columns=['Respondent', 'LastJob1', 'LastJob2', 'LastJob3'])
    item_details_html = candidate_details.to_html()
    return jsonify(item_details_html)

@app.route('/get_user_recommendations', methods=['GET'])
def get_user_recommendations():
    candidate_id = int(request.args.get('item_id'))
    top_k = 10
    cf_scores, metrics_dict, ids = kgat_wrapper.predict(candidate_id=candidate_id)
    zipped = zip(ids[1], cf_scores[0])
    sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
    top_k_recommendations = dict(sorted_[:top_k])
    top_job_ids = top_k_recommendations.keys()
    top_job_org_ids = lookup[lookup['id'].isin(top_job_ids)]['org_id']
    job_details = jobs[jobs['job_id'].isin(top_job_org_ids)].drop(columns=['job_id'])
    # job_details['score'] = top_k_recommendations.values()
    user_details_html = job_details.to_html()
    return jsonify(user_details_html)


# @app.route("/rc/<jobid>", methods=['GET'])
# def rc(jobid):
#     top_k = 10
#     cf_scores, metrics_dict, ids = kgat_wrapper.predict(job_id=jobid)
#     zipped = zip(ids[1], cf_scores[0])
#     sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
#     top_k_recommendations = dict(sorted_[:top_k])
#
#     job_org_id = lookup[lookup['id'] == int(jobid)]['org_id'].iloc[0]
#     job_details = jobs[jobs['job_id'] == job_org_id]
#
#     top_candidate_ids = top_k_recommendations.keys()
#     top_candidates_org_ids = lookup[lookup['id'].isin(top_candidate_ids)]['org_id']
#     top_candidates_org_ids = [int(x) for x in top_candidates_org_ids]
#     candidates_details = candidates[candidates['Respondent'].isin(top_candidates_org_ids)]
#     res = '<h3> Job Details </h3>'
#     res += job_details.to_html()
#     res += '<h3> Recommended candidates </h3>'
#     res += candidates_details.to_html()
#     return res

# @app.route("/rj/<candidateid>", methods=['GET'])
# def rj(candidateid):
#     top_k = 10
#     top_k_recommendations = []
#     cf_scores, metrics_dict, ids = kgat_wrapper.predict(candidate_id=candidateid)
#     zipped = zip(ids[1], cf_scores[0])
#     sorted_ = sorted(zipped, key=lambda x: x[1], reverse=True)
#     top_k_recommendations = dict(sorted_[:top_k])
#     return str(top_k_recommendations)


# @app.route("/check_similarity", methods=['GET'])
# def check_similarity():
#     return render_template("html/compare_form.html")

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
