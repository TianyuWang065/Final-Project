import numpy
import scipy.special

class NeuralNetwork:
    #初始化神经网络
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        
        #创建两个链接权重矩阵
        #正太分布的中心设定为0.0，使用下一层节点数的开方作为标准方差来初始化权重，即pow(self.hnodes,-0.5)，最后一个参数是numpy数组的形状大小
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        #学习率
        self.lr=learningrate
        
        #使用lambda来创建函数，这个函数接受了x，返回scipy.special.expit(x)，这就是S函数（激活函数）
        self.activation_function=lambda x:scipy.special.expit(x)
        
    
    #训练网络，反向传播误差
    #训练网络分两个部分：针对给定的训练样本输出，这与query()函数上所做内容没什么区别；将计算得到的输出与所需输出对比，使用差值来指导网络权重的更新
    def train(self,inputs_list,targets_list):#target_list目标值
        #将输入的列表转换为矩阵并且转置,数组的维度是2(2维数组表示矩阵)
        inputs=numpy.array(inputs_list,ndmin=2).T
        #将targets_list变成numpy数组（维度为2），也即是矩阵
        targets=numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #以上部分与query()部分使用完全相同的方式从输入层前馈信号到最终输出层
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
        #输出层输出误差为预期目标输出值与实际计算得到的输出值的差
        output_errors=targets-final_outputs
        #计算隐藏层节点反向传播的误差：隐藏层与输出层之间链接权重的转置点乘输出层输出误差，为隐藏层输出误差
        hidden_errors=numpy.dot(self.who.T,output_errors)
        
        
        #利用更新权重的公式进行计算，得到新的权重
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        
        pass
    
    #查询网络，计算输出
    def query(self,inputs_list):
        #将inputs_list变成numpy数组（维度为2），也即是矩阵
        inputs=numpy.array(inputs_list,ndmin=2).T
        
        #输入层与隐藏层链接权重矩阵点乘输入矩阵，得到隐藏层的输入矩阵
        hidden_inputs=numpy.dot(self.wih,inputs)
        #调用S函数，得到隐藏层的输出
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #隐藏层与输出层的链接权重点乘隐藏层的输出矩阵，得到输入层的输入矩阵
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #调用S函数，得到输出层的输出
        final_outputs=self.activation_function(final_inputs)
        
        #返回输出的输出矩阵
        return final_outputs
  