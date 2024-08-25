from flask import Flask, request, jsonify, send_from_directory
import torch
from model.hifi3dpp import ParametricFaceModel
import os
from utils.mesh_utils import write_mesh_obj
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/generate_mesh', methods=['POST'])
def generate_mesh():
    data = request.json
    exp_component = data['exp_component']
    change_value = data['change_value']

    # Setup for changing expression basis coefficient values
    coeffs = torch.load(data['model_coeffs_path'], map_location='cpu')
    coeffs_dict = split_coeff(coeffs)

    if exp_component < 1 or exp_component > 45:
        return jsonify({"error": "The expression component must be between 1 and 45"}), 400

    if change_value < -2.0 or change_value > 2.0:
        return jsonify({"error": "The amount changed must be between -2 and 2"}), 400

    coeffs_dict['exp'][0, exp_component - 1] += change_value

    # Initialise the parametric model for expression alteration
    face_model = ParametricFaceModel(fm_model_file='../topo_assets/hifi3dpp_model_info.mat', unwrap_info_file='../topo_assets/unwrap_1024_info.mat', device='cpu')
    output_path = data['output_path']
    mesh_name = f"{exp_component}_{change_value}.obj"
    saved_mesh = save_mesh(path=output_path, mesh_name=mesh_name, coeffs=coeffs, facemodel=face_model)

    return jsonify({"saved_mesh": saved_mesh})


@app.route('/download_mesh/<path:filename>', methods=['GET'])
def download_mesh(filename):
    processed_file = os.path.basename(filename)
    print(processed_file)
    file_path = os.path.join('../coeff_testing_framework', processed_file)
    print(file_path)
    if os.path.isfile(file_path):
        return send_from_directory(directory='../coeff_testing_framework', path=processed_file, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404


def split_coeff(coeffs):
    '''
    Split the estimated coeffs.
    '''

    if isinstance(coeffs, dict):
        coeffs = coeffs['coeffs']

    # Explicit indices for the coefficients
    id_coeffs = coeffs[:, :532] 
    exp_coeffs = coeffs[:, 532:532 + 45]
    tex_coeffs = coeffs[:, 532 + 45:532 + 45 + 439]
    angles = coeffs[:, 532 + 45 + 439:532 + 45 + 439 + 3]
    gammas = coeffs[:, 532 + 45 + 439 + 3:532 + 45 + 439 + 3 + 27]
    translations = coeffs[:, 532 + 45 + 439 + 3 + 27:]

    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


# Non-class implementation of save_mesh in ours_fit_model.py
def save_mesh(path, mesh_name, coeffs, facemodel):
    if isinstance(coeffs, dict):
        coeffs = coeffs['coeffs']
    coeffs_dict = facemodel.split_coeff(coeffs)
    
    pred_id_vertex, pred_exp_vertex, pred_alb_tex = facemodel.compute_for_mesh(coeffs_dict)
    
    exp_mesh_info = {
        'v': pred_exp_vertex.detach()[0].cpu().numpy(),
        'vt': pred_alb_tex.detach()[0].cpu().numpy(),
        'fv': facemodel.head_buf.cpu().numpy()
    }
    
    exp_mesh_path = os.path.join(path, f'{mesh_name[:-4]}_exp{mesh_name[-4:]}')
    write_mesh_obj(mesh_info=exp_mesh_info, file_path=exp_mesh_path)
    return exp_mesh_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)