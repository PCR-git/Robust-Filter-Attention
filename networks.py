import numpy as np
import torch
    
##########################################################################################
##########################################################################################

def apply_weight_masks(module, lambda_h_v, args):

    weight_mask_in = torch.zeros(args.d_e*2, args.d_e).to(args.device)
    weight_mask_in[0:2,0:2] = 1
    weight_mask_in[128:130, :] = 1

    weight_mask_out = torch.zeros(args.d_e, args.d_e*2).to(args.device)
    weight_mask_out[0:2,0:2] = 1
    weight_mask_out[:, 128:130] = 1

    bias_mask_in = torch.zeros(args.d_e*2).to(args.device)
    bias_mask_out = torch.zeros(args.d_e).to(args.device)

    with torch.no_grad():
        module.W_q.weight *= weight_mask_in
        module.W_k.weight *= weight_mask_in
        module.W_v.weight *= weight_mask_in
        module.W_o.weight *= weight_mask_out

        module.W_q.bias *= bias_mask_in
        module.W_k.bias *= bias_mask_in
        module.W_v.bias *= bias_mask_in
        module.W_o.bias *= bias_mask_out
        
        lambda_h_v[0] = lambda_h_v[0]*0 - 0.1
        lambda_h_v[1,:,0] = -1.0
        lambda_h_v[1,:,1] = 1.0
        lambda_h_v[1] = lambda_h_v[1]/torch.abs(lambda_h_v[1])
        
#         lambda_sigma_v = 0.0
#         lambda_eta_v = 1.0
#         lambda_gamma_v = 0.0

    print('Weight masking on.')

    return lambda_h_v

##########################################################################################
##########################################################################################

def apply_projection_mask(self):
        """
        Masks parameters of W_q, W_k, W_v, and W_o.
        Isolates the first 2 latent dimensions of the Real and Imaginary parts.
        """
        with torch.no_grad():
            # 1. Input Projections (W_q, W_k, W_v) -> Shape [2*d_out, d_in]
            for layer in [self.W_q, self.W_k, self.W_v]:
                d_total = self.d_k_total if layer != self.W_v else self.d_v_total
                
                weight_mask = torch.zeros_like(layer.weight)
                weight_mask[0:2, :] = 1.0                # Real part, first 2 dims
                weight_mask[d_total:d_total+2, :] = 1.0  # Imag part, first 2 dims
                layer.weight.mul_(weight_mask)
                
                if layer.bias is not None:
                    bias_mask = torch.zeros_like(layer.bias)
                    bias_mask[0:2] = 1.0
                    bias_mask[d_total:d_total+2] = 1.0
                    layer.bias.mul_(bias_mask)

            # 2. Output Projection (W_o) -> Shape [d_out, 2*d_in]
            # Here, the latent dimensions are in the columns (dim 1)
            d_latent_total = self.d_v_total 
            o_weight_mask = torch.zeros_like(self.W_o.weight)
            
            # Mask the columns: First 2 of Real, First 2 of Imag
            o_weight_mask[:, 0:2] = 1.0
            o_weight_mask[:, d_latent_total : d_latent_total + 2] = 1.0
            
            self.W_o.weight.mul_(o_weight_mask)
            
            # Note: W_o bias maps to d_e (real space), so we usually 
            # don't mask it unless you want to zero the entire output.
            # if self.W_o.bias is not None:
            #     self.W_o.bias.zero_()
            