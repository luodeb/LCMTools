#!/usr/bin/env python
# -*-coding:utf-8 -*-
# Authors:Debin/Gou Yaoping/
# requirements:
#     pip install python3-ghostscript
#     pip install numpy

import struct
import os
import numpy as np
import locale
import ghostscript
from scipy import signal
import time

class LCMConfig(object):
    # 输入data方式配置信息
    echot = "30"
    hzpppm = "1.27731e+02"
    nsize = "2048"

    ppmst = "4.0"
    ppmend = "0.2"
    deltat = '2.500e-04'
    B0 = '3.0'
    seq = 'PRESS'
    # 输入文件方式配置信息
    filraw:str = None

    def __str__(self):
        return "({}:{})".format(self.__class__.__name__, ",".join("{}={}".format(k, getattr(self, k)) for k in self.__dict__.keys()))

class RdaLcmodel(object):
    version = '2023/02/23 '
    def __init__(self, basisFile):
        self.basisFile = basisFile
        self.lcmConfig = None

    # 从文件中读取数据
    def load_file(self, baseFile, lcmConfig = LCMConfig()):
        '''
            只能加载.raw文件或者.rda文件
            lcmConfig.filraw: 如果不为空，那么就是在设置的地方生成文件，否则就是在baseFile的路径生成文件
        '''
        self.baseType = baseFile.split('.')[-1]
        self.__set_config(lcmConfig)
        if self.lcmConfig.filraw is not None:
            self.__fit_file(self.lcmConfig.filraw, self.basisFile)
        else:
            self.__fit_file(baseFile, self.basisFile)

        # 生成raw文件
        if self.baseType == 'rda':
            print('LCMTools[raw]: ' + self.filraw)
            self.__read_rda(baseFile)
        elif self.baseType == 'raw':
            # 读取raw文件
            self.__read_raw(baseFile)
            # 如果地址不一致，那么复制文件
            if self.filraw == baseFile:
                os.system(f'cp {baseFile} {self.filraw}')
        else:
            raise Exception('LCMTools: 未知的文件类型')

        self.__set_config(self.lcmConfig)

    def load_data(self, data, dataType='time' , lcmConfig = LCMConfig()):
        '''
            type:
            spec_real 是频域的实部数据;
            spec 是频域的复数据;
            time 是时域的复数据
        '''
        # 读取长度信息
        lcmConfig.nsize = len(data)
        # 加载配置信息
        self.__set_config(lcmConfig)

        # 生成文件名称
        self.__fit_file(lcmConfig.filraw, self.basisFile)

        # 生成raw文件
        self.__convert_data_raw(data, dataType)

    def __set_config(self, lcmConfig):
        
        self.lcmConfig = lcmConfig

        self.ctl_dict = {}

        # 经过一个简单的测试，这个部分是PPM的范围一般默认4.0-0.2不需要修改
        self.ctl_dict['ppmst'] = self.lcmConfig.ppmst
        self.ctl_dict['ppmend'] = self.lcmConfig.ppmend

        # 这个值是数据点数
        self.ctl_dict['nunfil'] = self.lcmConfig.nsize

        # hz每ppm
        self.ctl_dict['hzpppm'] = self.lcmConfig.hzpppm

        # echo time
        self.ctl_dict['echot'] = self.lcmConfig.echot

        # 采样时间
        self.ctl_dict['deltat'] = self.lcmConfig.deltat

        # 磁场强度
        self.ctl_dict['Bo'] = self.lcmConfig.B0

        # 序列
        self.seq = self.lcmConfig.seq

    def get_config(self):
        return self.lcmConfig

    def __fit_file(self, baseFile, basisFile):
        # 生成数据
        
        self.baseFile = ".".join(baseFile.split(".")[0:-1])

        # input
        self.filraw = self.baseFile + ".raw"
        self.filctrl = self.baseFile + ".control"
        self.filbas = basisFile

        # output
        self.filps = self.baseFile + ".ps"
        self.filcsv = self.baseFile + ".csv"
        self.filpdf = self.baseFile + ".pdf"

        self.title = self.baseFile.split('/')[-1]

    def __read_rda(self,rdaFile: str ):
        # 将rda文件转换为raw文件，并且读取配置信息
        with open(rdaFile, 'rb') as file:
            rdaText = file.read()

        headtext = rdaText[rdaText.find(b'>>> Begin of header <<<') + len(
            b'>>> Begin of header <<<') + 2: rdaText.find(b'>>> End of header <<<') - 2].decode()
        head_dict = dict([line.split(": ", 1)
                         for line in headtext.split("\n")])

        bytext = rdaText[rdaText.find(
            b'End of header <<<') + len(b'End of header <<<')+2:]

        self.ctl_dict['echot'] = head_dict['TE']
        self.lcmConfig.echot = head_dict['TE']

        self.ctl_dict['hzpppm'] = '%e' % float(head_dict['MRFrequency'])
        self.lcmConfig.hzpppm = '%e' % float(head_dict['MRFrequency'])

        self.ctl_dict['nunfil'] = "%d" % int(len(bytext)/16)
        self.lcmConfig.nsize = "%d" % int(len(bytext)/16)

        self.ctl_dict['ID'] = head_dict['PatientID'].strip()

        self.ctl_dict['deltat'] = '%e' % (
            float(head_dict['DwellTime'])/1000000.0)
        self.lcmConfig.deltat = '%e' % (
            float(head_dict['DwellTime'])/1000000.0)
        
        self.ctl_dict['Bo'] = head_dict['MagneticFieldStrength']
        self.lcmConfig.B0 = head_dict['MagneticFieldStrength']

        # 此处是预处理生成header数据的部分
        headers = (f" $SEQPAR\n echot= {self.ctl_dict['echot']}\n seq= 'PRESS'\n hzpppm= {self.ctl_dict['hzpppm']}\n $END\n "
                   f"$NMID\n id='{self.ctl_dict['ID']}', fmtdat='(2E15.6)'\n volume=8.000e+00\n tramp=1.0\n $END\n  ")

        data = struct.unpack('%dd' % int(len(bytext)/8), bytext)

        wrdata = ""
        self.time_data = np.zeros(int(len(data)/2),dtype=complex)
        for i in range(int(len(data)/2)):
            wrdata += "{:>13.6e}  {:>13.6e}\n  ".format(data[2*i], data[2*i+1])
            self.time_data[i] = float(data[2*i]) - 1j*float(data[2*i+1])
        with open(self.filraw, 'w') as f:
            f.write(headers + wrdata)

        self.spec_data = np.fft.fftshift(np.fft.fft(self.time_data))

    def __convert_data_raw(self, data, dtype='time'):
        '''
            type:
            spec_real 是频域的实部数据;
            spec 是频域的复数据;
            time 是时域的复数据
        '''
        fid_mod = np.array([])
        if dtype == 'time':
            fid_mod = data
        elif dtype == 'spec':
            fid_mod = np.fft.ifft(np.fft.ifftshift(data))
        elif dtype == 'spec_real':
            specs = np.conj(signal.hilbert(data))
            fid_mod = np.fft.ifft(np.fft.ifftshift(specs))

        self.time_data = fid_mod
        self.spec_time = np.fft.fftshift(np.fft.fft(fid_mod))

        data = np.zeros(int(len(fid_mod)*2))
        data[::2] = fid_mod.real
        data[1::2] = -1*fid_mod.imag
        wrdata = ""
        for i in range(int(len(data) / 2)):
            wrdata += "{:>13.6e}  {:>13.6e}\n  ".format(data[2*i], data[2*i+1])

        headers = f"$SEQPAR\n echot= {self.lcmConfig.echot}\n seq= 'PRESS'\n hzpppm= {self.lcmConfig.hzpppm}\n $END\n $NMID\n id='MR341785 ', " \
            "fmtdat='(2E15.6)'\n volume=8.000e+00\n tramp=1.0\n $END\n  "
        with open(self.filraw, 'w') as f:
            f.write(headers + wrdata)

    def __read_raw(self,filraw:str):
        # 读取raw文件
        with open(filraw, 'rb') as file:
            rdaText = file.read().decode()
        byindex = rdaText.find('$END',rdaText.find('$END') + 1)+4
        headText = rdaText[:byindex]
        dataText = rdaText[byindex:]
        data  = dataText.replace('\n','').split()
        self.time_data = np.zeros(len(data) // 2,dtype=complex)
        for i in range(len(data)//2):
            self.time_data[i] = float(data[2*i]) - 1j * float(data[2*i+1])

        self.spec_data = np.fft.fftshift(np.fft.fft(self.time_data))

        for head in headText.split('\n'):
            # 如果head中包含echot
            if 'echot' in head:
                self.lcmConfig.echot = head.split('=')[1].replace(' ','')
            elif 'hzpppm' in head:
                self.lcmConfig.hzpppm = head.split('=')[1].replace(' ','')
            elif 'seq' in head:
                self.lcmConfig.seq = head.split('=')[1].replace(' ','').replace("'",'')
            elif 'NumberOfPoints' in head:
                self.lcmConfig.nsize = head.split('=')[1].replace(' ','')
            elif 'dwellTime' in head:
                self.lcmConfig.deltat = head.split('=')[1].replace(' ','')

    def get_data(self, dtype='time'):
        if dtype == 'time':
            return self.time_data
        elif dtype == 'spec':
            return self.spec_data
        else:
            assert False, 'dtype must be time or spec'

    def __gen_control(self):
        # 生成control文件
        # control file
        control_text = (f" $LCMODL\n title= '{self.title}'\n "
                        f"ppmst= {self.ctl_dict['ppmst']}\n ppmend= {self.ctl_dict['ppmend']}\n "
                        f"nunfil= {self.ctl_dict['nunfil']}\n key= 210387309\n hzpppm= {self.ctl_dict['hzpppm']}\n "
                        f"filraw= '{self.filraw}'\n filps= '{self.filps}'\n filbas= '{self.filbas}'\n filcsv= '{self.filcsv}'\n "
                        f"lcsv = 11\n echot= {self.ctl_dict['echot']}\n deltat= {self.ctl_dict['deltat']}\n $END")

        with open(self.filctrl, 'w') as control:
            control.write(control_text)

    def __convert_ps_pdf(self):
        # 将PostScript文件转换为pdf文件
        args = [
            "ps2pdf",  # actual value doesn't matter
            "-dNOPAUSE", "-dBATCH", "-dSAFER",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={self.filpdf}",
            "-c", ".setpdfwrite",
            "-f",  self.filps
        ]

        # arguments have to be bytes, encode them
        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]

        print('\nLCMTools[pdf]: ',*args)
        ghostscript.Ghostscript(*args)

    def __run_command(self):
        # 运行lcmodel命令
        self.lc_command = '~/.lcmodel/bin/lcmodel < ' + self.filctrl
        os.system(self.lc_command)

    def run_lcmodel(self, delTemp=False):
        print('LCMTools[version]: ', self.version)

        # 打印现在的时间
        print('LCMTools[time]: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        print('LCMTools[config]: ', self.lcmConfig)
        # 依次执行相应的函数
        print('LCMTools[control]: ' + self.filctrl)
        self.__gen_control()

        self.lc_command = '~/.lcmodel/bin/lcmodel < ' + self.filctrl
        print('LCMTools[lcmodel]: ' + self.lc_command)
        self.__run_command()

        self.__convert_ps_pdf()
        if delTemp:
            self.clean_temp()

    def clean_temp(self, fileList:list=['control', 'raw', 'ps', 'csv']):
        # 删除不需要的文件
        for file in fileList:
            try:
                os.remove(self.baseFile + '.'+file)
            except:
                print('LCMTools[error]: ' + self.baseFile + '.'+file + ' not found')

if __name__ == "__main__":
    # 读取raw文件
    filepath = "/data2/ldb/objects/fida/out0222.raw"
    lcm = RdaLcmodel("/data2/ldb/objects/lcmodel/basis/3t/press_te30_3t_v3.basis")
    lcm.load_file(filepath)
    lcm.run_lcmodel()
    spec_data = lcm.get_data('spec')

    # 直接将频域的实部转换为raw文件
    lcm_config = LCMConfig()
    lcm_config.echot = '30.0'
    lcm_config.filraw = '/data2/ldb/objects/fida/out0223.raw'
    lcm2 = RdaLcmodel("/data2/ldb/objects/lcmodel/basis/3t/press_te30_3t_v3.basis")
    lcm2.load_data(spec_data.real,'spec_real',lcm_config)
    lcm2.run_lcmodel()
