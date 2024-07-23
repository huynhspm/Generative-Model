import torch
import pyrootutils
import torch.nn as nn

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels,  out_channels, base_channels, number_unet, 
                 conv_layer, norm_layer, activate_layer, transpconv_layer, 
                 conv_kwargs, norm_kwargs, activate_kwargs, transpconv_kwargs):
        
        super().__init__()

        # number of unet
        self.number_unet = number_unet
        # name of convolution layer
        self.conv_layer = conv_layer if type(conv_layer) is not str else getattr(nn, conv_layer)
        # name of normalization layer
        self.norm_layer = norm_layer if type(norm_layer) is not str else getattr(nn, norm_layer)
        # name of activation function layer
        self.activate_layer = activate_layer if type(activate_layer) is not str else getattr(nn, activate_layer)
        # name of transposed convolution layer
        self.transpconv_layer = transpconv_layer if type(transpconv_layer) is not str else getattr(nn, transpconv_layer)
        
        # parameters of convolution layer
        self.conv_kwargs = conv_kwargs
        # parameters of normalization layer
        self.norm_kwargs = norm_kwargs
        # parameters of activation function layer
        self.activate_kwargs = activate_kwargs
        # parameters of transposed convolution layer
        self.transpconv_kwargs = transpconv_kwargs


        # input convolution layer to base_channels 
        self.first_conv = self.get_conv_block(in_channels, base_channels, have_pool=False)

        # down convolution modules
        self.down_conv_modules = [None] * number_unet
        # up convolution modules
        self.up_modules = [[None] * (i + 1) for i in range(number_unet)]
        # up convolution modules
        self.up_conv_modules = [[None] * (i + 1) for i in range(number_unet)]
        
       
        # # number of channels at each level
        self.channels = [base_channels] + [base_channels * (1 << (i + 1)) for i in range(number_unet)]
            
        # initial modules for unetplusplus
        for i in range(number_unet):
            # i-th unet

            # i-th down convolution layer of all unets
            self.down_conv_modules[i] = self.get_conv_block(self.channels[i], self.channels[i + 1])

            # up layers of i-th unet
            for j in range(i + 1):
                # sum of channels after concat
                in_channels_conv = (j + 2) * self.channels[i - j]
                
                # j-th up layer of i-th unet
                self.up_modules[i][j], self.up_conv_modules[i][j] = \
                    self.get_up_block(self.channels[i + 1 - j], self.channels[i - j], in_channels_conv)            
        
            self.up_modules[i] = nn.ModuleList(self.up_modules[i])
            self.up_conv_modules[i] = nn.ModuleList(self.up_conv_modules[i])
        
        self.down_conv_modules = nn.ModuleList(self.down_conv_modules)
        self.up_modules = nn.ModuleList(self.up_modules)
        self.up_conv_modules = nn.ModuleList(self.up_conv_modules)
        
        # output convolution to out_channels
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, input):
        x = [[None] * (i + 1) for i in range(self.number_unet + 1)]
        
        # input convolution layer to base_channels 
        x[0][0] = self.first_conv(input)
        
        for i in range(self.number_unet):
            # i-th down layer of all unets
            x[i + 1][0] = self.down_conv_modules[i](x[i][0])
            
            # up layers of i-th unet
            for j in range(i + 1):
                # j-th up layer of i-th unet

                up_element = self.up_modules[i][j](x[i + 1][j])
                cat_elements = [up_element]
                for k in range(j + 1):
                    cat_elements.append(x[i - k][j - k])
                
                # up convolution after concat
                x[i + 1][j + 1] = self.up_conv_modules[i][j](torch.cat(cat_elements, dim=1))
        
        output = self.output_conv(x[self.number_unet][self.number_unet])
        return output        
                
    def get_conv_block(self, in_channels, out_channels, have_pool=True):
        if not have_pool:
            stride = 1
        else:
            stride = 2
            
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels, stride = stride, **self.conv_kwargs),
            self.norm_layer(out_channels, **self.norm_kwargs),
            self.activate_layer(**self.activate_kwargs),
            self.conv_layer(out_channels, out_channels, stride = 1, **self.conv_kwargs),
            self.norm_layer(out_channels, **self.norm_kwargs),
            self.activate_layer(**self.activate_kwargs),
        )
    
    def get_up_block(self, in_channels, out_channels, in_channels_conv):
        up = self.transpconv_layer(in_channels, out_channels, **self.transpconv_kwargs)
        up_conv = nn.Sequential(
            self.get_conv_block(in_channels_conv, out_channels, have_pool=False),
            self.get_conv_block(out_channels, out_channels, have_pool=False)
        )
        return up, up_conv


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(root / "configs" / "model" / "unet" / "net")
    print("root: ", root)

    @hydra.main(version_base=None, 
                config_path=config_path, 
                config_name="unet_plus_plus.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        unet_plus_plus: UNetPlusPlus = hydra.utils.instantiate(cfg)
        image = torch.randn(2, 1, 240, 240)
        
        logits = unet_plus_plus(image)
        
        print('***** UNet Plus Plus*****')
        print('Input:', image.shape)
        print('Output:', logits.shape)
    
    main()