import os
import cv2
import time
import itertools
from dataset import custom_reader
from network import *
from utils import *
from glob import glob
from work.hapi.vision.transform import transforms
import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt


class UGATIT(object):
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'
        
        self.result_dir = args.result_dir   # 保存结果的路径
        self.dataset = args.dataset         # dataset名称

        self.iteration = args.iteration     # 训练迭代次数
        self.decay_flag = args.decay_flag   # 衰减标志

        self.batch_size = args.batch_size   
        self.print_freq = args.print_freq   # 图像打印频率
        self.save_freq = args.save_freq     # 模型保存频率

        self.lr = args.lr                   # 学习率
        self.weight_decay = args.weight_decay   # 权重衰减率
        self.ch = args.ch                   # 每层基础通道数

        """ Weight """
        self.adv_weight = args.adv_weight           # GAN权重
        self.cycle_weight = args.cycle_weight       # Cycle权重
        self.identity_weight = args.identity_weight # 一致性权重
        self.cam_weight = args.cam_weight           # CAM权重

        """ Generator """
        self.n_res = args.n_res         # 残差块数量

        """ Discriminator """
        self.n_dis = args.n_dis         # 判别器层数

        self.img_size = args.img_size   # 图像大小 -> 256
        self.img_ch = args.img_ch       # 图像通道数 -> 3

        self.device = args.device       # 运行设备，cpu / gpu
        self.benchmark_flag = args.benchmark_flag   # 
        self.resume = args.resume       # 恢复标志

        # if torch.backends.cudnn.enabled and self.benchmark_flag:
        #     print('set benchmark !')
        #     torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    #    Model
    ##################################################################################
    def build_model(self):
        """ DataLoader"""
        # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        #     transforms.Resize((self.img_size+30, self.img_size+30)),
        #     transforms.RandomCrop(self.img_size),
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((self.img_size, self.img_size)),
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])

        self.trainA_loader = paddle.batch(custom_reader('/home/aistudio/dataset/trainA'), batch_size=self.batch_size, drop_last=True)
        self.trainB_loader = paddle.batch(custom_reader('/home/aistudio/dataset/trainB'), batch_size=self.batch_size, drop_last=True)
        self.testA_loader = paddle.batch(custom_reader('/home/aistudio/dataset/testA'), batch_size=1, drop_last=True)
        self.testB_loader = paddle.batch(custom_reader('/home/aistudio/dataset/testB'), batch_size=1, drop_last=True)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = fluid.dygraph.L1Loss()
        self.MSE_loss = fluid.dygraph.MSELoss()
        self.BCE_loss = fluid.dygraph.BCELoss()

        """ Trainer """
        self.G_optim = fluid.optimizer.AdamOptimizer(parameter_list=self.genA2B.parameters() + self.genB2A.parameters(),
                                                     learning_rate=self.lr, 
                                                     beta1=0.5, beta2=0.999,
                                                     regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
                                                    )
        self.D_optim = fluid.optimizer.AdamOptimizer(parameter_list=self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters(),
                                                     learning_rate=self.lr, 
                                                     beta1=0.5, beta2=0.999,
                                                     regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
                                                    )

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        start_iter = 1
        # model实例化后，指定为 train 状态
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        if self.resume:
            model_list = glob(os.path.join('/home/aistudio/result', 'model', '*.pdparams'))    # 查找符合特定规则的文件路径名
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join('/home/aistudio/result', 'model'), start_iter)
                print(" [*] Load SUCCESS")
                # if self.decay_flag and start_iter > (self.iteration // 2):
                #     print(self.G_optim)
                #     self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                #     self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('\n training start ! \n')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            # if self.decay_flag and step > (self.iteration // 2):
            #     self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
            #     self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                
            try:
                real_A = next(trainA_iter)
                real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)
            except:
                trainA_iter = iter(self.trainA_loader())      # iter生成迭代器，python基本语法
                real_A = next(trainA_iter)
                real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)

            try:
                real_B = next(trainB_iter)
                real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)
            except:
                trainB_iter = iter(self.trainB_loader())
                real_B = next(trainB_iter)
                real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)

            real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)


            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            # Update D
            self.D_optim.clear_gradients()

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, fluid.layers.zeros_like(fake_GA_logit))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, fluid.layers.zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, fluid.layers.zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, fluid.layers.zeros_like(fake_GB_logit))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, fluid.layers.zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, fluid.layers.zeros_like(fake_LB_logit))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, fluid.layers.zeros_like(fake_LB_cam_logit))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B


            Discriminator_loss.backward()
            self.D_optim.minimize(Discriminator_loss)


            # Update G
            self.G_optim.clear_gradients()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.layers.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.layers.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.layers.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.layers.ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.layers.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.layers.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.layers.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.layers.ones_like(fake_LB_cam_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fluid.layers.sigmoid(x=fake_B2A_cam_logit), fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(x=fake_A2A_cam_logit), fluid.layers.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = self.BCE_loss(fluid.layers.sigmoid(x=fake_A2B_cam_logit), fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(x=fake_B2B_cam_logit), fluid.layers.zeros_like(fake_B2B_cam_logit))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.minimize(Generator_loss)

            # # clip parameter of AdaILN and ILN, applied after optimizer step
            # self.genA2B.apply(self.Rho_clipper)
            # self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            
            # 打印训练效果
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A = next(trainA_iter)
                        real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)
                    except:
                        trainA_iter = iter(self.trainA_loader())
                        real_A = next(trainA_iter)
                        real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)

                    try:
                        real_B = next(trainB_iter)
                        real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)
                    except:
                        trainB_iter = iter(self.trainB_loader())
                        real_B = next(trainB_iter)
                        real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)
                        
                    real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)


                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                            cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                            cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                            cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                            cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                            cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                            cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                            RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)


                for _ in range(test_sample_num):
                    try:
                        real_A = next(testA_iter)
                        real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)
                    except:
                        testA_iter = iter(self.testA_loader())
                        real_A = next(testA_iter)
                        real_A = np.array([x.transpose(2, 0, 1) for x in real_A], np.float32)

                    try:
                        real_B = next(testB_iter)
                        real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)
                    except:
                        testB_iter = iter(self.testB_loader())
                        real_B = next(testB_iter)
                        real_B = np.array([x.transpose(2, 0, 1) for x in real_B], np.float32)
                        
                    real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join('/home/aistudio/result', 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join('/home/aistudio/result', 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            
            if step % self.save_freq == 0:
                self.save(os.path.join('/home/aistudio/result', 'model'), step)



    def save(self, dir, step):
        model_path = dir + '/'

        fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(model_path + 'genA2B_%07d' % step))
        fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(model_path + 'genB2A_%07d' % step))
        fluid.save_dygraph(self.disGA.state_dict(), os.path.join(model_path + 'disGA_%07d' % step))
        fluid.save_dygraph(self.disGB.state_dict(), os.path.join(model_path + 'disGB_%07d' % step))
        fluid.save_dygraph(self.disLA.state_dict(), os.path.join(model_path + 'disLA_%07d' % step))
        fluid.save_dygraph(self.disLB.state_dict(), os.path.join(model_path + 'disLB_%07d' % step))
        # fluid.save_dygraph(self.G_optim.state_dict(), os.path.join(model_path + 'G_optim_%07d' % step))
        # fluid.save_dygraph(self.D_optim.state_dict(), os.path.join(model_path + 'D_optim_%07d' % step))


    def load(self, dir, step):
        model_path = dir + '/'

        genA2B_para, _ = fluid.load_dygraph(os.path.join(model_path + 'genA2B_%07d' % step))
        genB2A_para, _ = fluid.load_dygraph(os.path.join(model_path + 'genB2A_%07d' % step))
        disGA_para, _ = fluid.load_dygraph(os.path.join(model_path + 'disGA_%07d' % step))
        disGB_para, _ = fluid.load_dygraph(os.path.join(model_path + 'disGB_%07d' % step))
        disLA_para, _ = fluid.load_dygraph(os.path.join(model_path + 'disLA_%07d' % step))
        disLB_para, _ = fluid.load_dygraph(os.path.join(model_path + 'disLB_%07d' % step))

        self.genA2B.load_dict(genA2B_para)
        self.genB2A.load_dict(genB2A_para)
        self.disGA.load_dict(disGA_para)
        self.disGB.load_dict(disGB_para)
        self.disLA.load_dict(disLA_para)
        self.disLB.load_dict(disLB_para)

    def test(self):
        model_list = glob(os.path.join('/home/aistudio/result', 'model', '*.pdparams'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join('/home/aistudio/result', 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, real_A in enumerate(self.testA_loader()):
            real_A = fluid.dygraph.to_variable(np.array([x.transpose(2, 0, 1) for x in real_A], np.float32))

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join('/home/aistudio/result', 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
            print('testA %d finish !!' % (n+1))

        for n, real_B in enumerate(self.testB_loader()):
            real_B = fluid.dygraph.to_variable(np.array([x.transpose(2, 0, 1) for x in real_B], np.float32))

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join('/home/aistudio/result', 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
            print('testB %d finish !!' % (n+1))

