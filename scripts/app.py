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
    change_values = data['change_values']

    # Setup for changing expression basis coefficient values
    coeffs = torch.load(data['model_coeffs_path'], map_location='cpu')
    coeffs_dict = split_coeff(coeffs)

    open("../emotion.txt", 'w').close()
    with open("../emotion.txt", "a") as file:
        for i, value in enumerate(change_values):
            coeffs_dict['exp'][0, i] += value
            file.write(str(i+1) + " " + str(value) + '\n')

    # Initialise the parametric model for expression alteration
    face_model = ParametricFaceModel(fm_model_file='../topo_assets/hifi3dpp_model_info.mat', unwrap_info_file='../topo_assets/unwrap_1024_info.mat', device='cpu')
    output_path = data['output_path']
    mesh_name = "emotion.obj"
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

# Non-class based implementation of split_coeff in hifi3dpp.py
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


# Non-class implementation of save_mesh in ours_fit_model.py in the original repository:  https://github.com/csbhr/FFHQ-UV

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