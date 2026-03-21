import torch
import torch.nn as nn
import numpy


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, qmin, qmax):
        x = input * scale
        x = torch.round(x)
        x = torch.clamp(x, qmin, qmax)
        x = x / scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class FP8Convert():
    def __init__(self, model):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights
        
        self.target_modules = [] # this will contain the list of modules to be modified 
        # ADDENDUM : Target_modules is list of layers basically

        self.model = model # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method
        self.save_params()
    
        ### (2) Convert the weights in the model to FP8, by iterating through the list of target modules and overwrite the values with their binary version
        for target_module in self.target_modules:
            w = target_module.data

            # Compute symmetric scaling factor
            max_val = w.abs().max()
            if max_val == 0:
                scale = 1.0
            else:
                scale = 127.0 / max_val

            # Scale → Round → Clamp
            w_q = torch.round(w * scale)
            w_q = torch.clamp(w_q, -127, 127)

            # De-scale back to float domain
            w_q = w_q / scale

            target_module.data.copy_(w_q)
        
    def restore(self):

        ### restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    def clip(self):

        ## To be completed 
        ## Clip all parameters to the range [-127,127] 

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(torch.clamp(self.target_modules[index].data, -127, 127))


    def forward(self,x):
        ### This function is used so that the model can be used while training
        out = x

        for layer in self.model.children():
            out = layer(out)

            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # Compute dynamic scale (same idea as weights)
                max_val = out.abs().max()
                if max_val == 0:
                    scale = 1.0
                else:
                    scale = 127.0 / max_val

                out = QuantizeSTE.apply(out, scale, -127, 127)

        return out


class FP6Convert():
    def __init__(self, model):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights
        
        self.target_modules = [] # this will contain the list of modules to be modified 
        # ADDENDUM : Target_modules is list of layers basically

        self.model = model # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method
        self.save_params()
    
        ### (2) Convert the weights in the model to FP8, by iterating through the list of target modules and overwrite the values with their binary version
        for target_module in self.target_modules:
            w = target_module.data

            # Compute symmetric scaling factor
            max_val = w.abs().max()
            if max_val == 0:
                scale = 1.0
            else:
                scale = 32.0 / max_val

            # Scale → Round → Clamp
            w_q = torch.round(w * scale)
            w_q = torch.clamp(w_q, -32.0, 32.0)

            # De-scale back to float domain
            w_q = w_q / scale

            target_module.data.copy_(w_q)
        
    def restore(self):

        ### restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    def clip(self):

        ## To be completed 
        ## Clip all parameters to the range [-127,127] 

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(torch.clamp(self.target_modules[index].data, -32.0, 32.0))


    def forward(self, x):
        out = x

        for layer in self.model.children():
            out = layer(out)

            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                max_val = out.abs().max()
                if max_val == 0:
                    scale = 1.0
                else:
                    scale = 32.0 / max_val

                out = QuantizeSTE.apply(out, scale, -32.0, 32.0)

        return out