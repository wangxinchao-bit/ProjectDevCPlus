


import cv2
import time
import numpy as np
import torch.functional as F
import torch.nn as nn
import os
import glob
import numpy as np 
import onnxruntime as ort
import onnx
import torch  

torch.cuda.empty_cache()
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def convert_onnx(model, output_folder,onnx_name, device ="cuda:0",is_dynamic_batches=True): 
    """
        Convert the pth or pt model to Onnx Model
        
        Args:
        - model: The network Model Object 
        - output_folder: The onnx model  output path
        - is_dynamic_batches: A boolean indicating whether dynamic batches are used
        Returns:
            None
    """
    model.eval() 

    inputTensor = torch.randn(1, 3, 1080, 1920, requires_grad=False)
    os.makedirs(output_folder) if not os.path.exists(output_folder) else None

    output_name = os.path.join(output_folder, onnx_name)
    dynamic_params = None
    if is_dynamic_batches:
        dynamic_params = {  
            "input": {0: 'batch_size', 2 : 'in_width', 3: 'in_height'},   # 输入规定了三个动态轴，第一个是batch_size第二个是宽度，第三个是高度
            "output": {0: 'batch_size', 2: 'out_width', 3:'out_height'}
        } 
        
    # module = torch.jit.trace(model, inputTensor)
    # model_script = torch.jit.script(model)
    
    torch.onnx.export(model,            # model being run 
        inputTensor,            # model input (or a tuple for multiple inputs) 
        output_name,                    # where to save the model  
        export_params= True,            # store the trained parameter weights inside the model file 
        opset_version=16,               # the ONNX version to export the model to 
        do_constant_folding=True,       # whether to execute constant folding for optimization 
        input_names=["input"],  
        output_names=["output"],  
        # dynamic_axes = dynamic_params
    )


def test_onnx(onnx_model, input_paths, save_path):
    """
    Args:
        test_onnx: onnx model path
        input_path: test img path
        save_path: the save path for result
    
    """
    ort_session = ort.InferenceSession(onnx_model)

    images = []
    for path in input_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)  # Resize if necessary
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))  # 首先是通道转换转为RGB操作然后，然后转换为CHW (cv中的数据是HWC)
        images.append(img)

    batch_input = np.stack(images, axis=0)
    # Run inference
    start = time.time()
    outputs = ort_session.run(None, {"input": batch_input})
    end = time.time()
    print("Spend Time: ",end-start)


   # Save the output image
    for i, output in enumerate(outputs[0]):
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
        print(output)
        cv2.imwrite(os.path.join(save_path, f'res{i}.png'), output)

if __name__ == "__main__":
    
    from repnet import repnet
    
    model_path = "repnet_x2.pth"
    model = repnet(upscale=2)
    model_info = torch.load(model_path)

    model.load_state_dict(model_info, strict=True)
    output_folder = './onnx_res/' 
    print("开始导出Onnx")
    convert_onnx(model, output_folder,"retnet_0314.onnx",is_dynamic_batches=True)
    print("导出Onnx 结束")
    
    # # ###################Test the converted model #################
    onnx_model = 'onnx_res/retnet_0314.onnx'
    # input_path = '/home/wxcwxc/wxcpython/rt4ksr/res_aoto/imgs'
    # input_paths = sorted(glob.glob(os.path.join(input_path, '*')))
    input_paths =["/home/wxcwxc/wxcpython/rt4ksr_aoto/0001x2.png"]
    save_path = './res'
    test_onnx(onnx_model, input_paths, save_path)
