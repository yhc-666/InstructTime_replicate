"""
作用: 实现TStokenizer模型的训练和评估流程控制
输入: 
    - 模型实例
    - 数据加载器(训练和测试)
    - 训练参数配置
输出: 
    - 训练日志和指标
    - 保存的模型检查点
示例用法:
    trainer = Trainer(
        args=args,                 # 参数配置
        model=model,               # TStokenizer模型实例
        train_loader=train_loader, # 训练数据加载器
        val_loader=val_loader,     # 验证数据加载器
        verbose=True               # 是否显示详细信息
    )
    trainer.train()  # 开始训练流程
"""

import time
import torch
from tqdm import tqdm
from loss import MSE
from torch.optim.lr_scheduler import LambdaLR

class Trainer():
    """
    功能: 训练管理器，控制VQVAE模型的训练和评估流程
    """
    def __init__(self, args, model, train_loader, val_loader, verbose=False):
        """
        功能: 初始化训练管理器
        输入:
            - args: 参数配置对象
            - model: VQVAE模型实例
            - train_loader: 训练数据加载器
            - val_loader: 验证数据加载器
            - verbose: 是否显示详细信息
        """
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_decay = args.lr_decay_rate  # 学习率衰减比例
        self.lr_decay_steps = args.lr_decay_steps  # 学习率衰减步数
        self.weight_decay = args.weight_decay  # 权重衰减率
        self.model_name = self.model.get_name()
        self.print_process(self.model_name)

        # 初始化MSE损失函数
        self.cr = MSE(self.model)

        self.num_epoch = args.num_epoch  # 训练轮数
        self.eval_per_steps = args.eval_per_steps  # 每多少步执行一次评估
        self.save_path = args.save_path  # 模型保存路径
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0  # 当前训练步数
        self.best_metric = -1e9  # 最佳评估指标值
        self.metric = 'mse'  # 使用MSE作为评估指标

    def train(self):
        """
        功能: 执行完整的训练流程
        返回值:
            - 最佳评估指标值
        流程:
            1. 初始化优化器和学习率调度器
            2. 遍历训练轮数
            3. 每轮结束后记录损失和时间
        """
        # 初始化Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        # 初始化学习率调度器，按步数衰减
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        
        # 遍历训练轮数
        for epoch in range(self.num_epoch):
            # 训练一个epoch并获取平均损失和耗时
            loss_epoch, time_cost = self._train_one_epoch()
            
            # 记录训练日志
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Basic Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Basic Model train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        
        # 打印最佳指标值
        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        """
        功能: 训练一个epoch
        返回值:
            - loss_sum/idx: 平均损失值
            - time_cost: 训练耗时
        流程:
            1. 记录开始时间
            2. 遍历训练数据批次
            3. 前向传播计算损失
            4. 反向传播更新参数
            5. 按计划调整学习率
            6. 定期评估模型并保存最佳模型
        """
        t0 = time.perf_counter()
        self.model.train()
        # 使用tqdm显示进度条(如果verbose为True)
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            # 清空梯度
            self.optimizer.zero_grad()
            if isinstance(batch, (list, tuple)):
                seqs, mask, _ = batch
            else:
                seqs, mask = batch, None
            loss = self.cr.compute(seqs, mask)
            loss_sum += loss.item()

            # 反向传播
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            # 更新参数
            self.optimizer.step()

            # 计步
            self.step += 1
            # 按计划衰减学习率
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
                
            # 定期评估模型
            if self.step % self.eval_per_steps == 0:
                # 评估模型并获取指标
                metric = self.eval_model_vqvae()
                self.print_process(metric)
                
                # 记录评估结果
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                
                # 如果当前指标优于历史最佳，保存模型
                if metric[self.metric] >= self.best_metric:
                    self.model.eval()
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric[self.metric]
                # 恢复训练模式
                self.model.train()

        return loss_sum / idx, time.perf_counter() - t0

    def eval_model_vqvae(self):
        """
        功能: 评估VQVAE模型性能
        返回值:
            - metrics: 包含评估指标的字典
        流程:
            1. 将模型设为评估模式
            2. 遍历测试数据批次
            3. 计算重构损失(MSE)
            4. 返回平均指标
        注意:
            - MSE取负值是因为训练器会选择较大的指标值作为"更好"
        """
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if self.verbose else self.val_loader
        metrics = {'mse': 0}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                if isinstance(batch, (list, tuple)):
                    seqs, mask, _ = batch
                else:
                    seqs, mask = batch, None
                mse = self.cr.compute(seqs, mask)
                # 注意这里取负值，因为评估选择较大的指标值作为"更好"
                metrics['mse'] -= mse
        # 计算平均指标
        metrics['mse'] /= idx
        return metrics
    
    def print_process(self, *x):
        """
        功能: 根据verbose标志打印信息
        输入:
            - *x: 要打印的内容
        """
        if self.verbose:
            print(*x)
