#/usr/bin/python2.7
#coding:utf-8
import numpy as np
import csv
global staut_list
def preHandle(inputfile, outputfile):
    print(inputfile, outputfile)
    source_file = inputfile
    handled_file = outputfile
    data_to_flie=open(handled_file, 'w')
    with (open(source_file,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        count=0
        for i in csv_reader:
            # print i
            temp_line=np.array(i)
            # temp_line[1]=handleProtocol(i)         #将源文件行中3种协议类型转换成数字标识
            # temp_line[2]=handleService(i)          #将源文件行中70种网络服务类型转换成数字标识
            # temp_line[3]=handleFlag(i)             #将源文件行中11种网络连接状态转换成数字标识
            

            handled_protocol = np.array(handleProtocol(i))
            handled_service = np.array(handleService(i))
            handled_flag = np.array(handleFlag(i))

            temp_line[41]=handleLabel(i)          #将源文件行中23种攻击类型转换成数字标识


            temp_line = np.delete(temp_line, [1,2,3])   # 删除这三列，它们已经转为[1,0,0,...,0]的形式了
            # print(temp_line[0])
            temp_line = np.concatenate((temp_line,handled_protocol, handled_service, handled_flag),axis=0)

            

            csv_writer.writerow(temp_line)
            #print count,'staus:',temp_line[1],temp_line[2],temp_line[3],temp_line[41]
            count+=1
            # return 1
        print('processed %d lines data' % count)
        data_to_flie.close()
def find_index(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

def handleProtocol(input):
    handled_list = []
    protoclo_list=['tcp','udp','icmp']
    for i in range(len(protoclo_list)):
        if input[1] == protoclo_list[i]:
            handled_list.append(1)
        else:
            handled_list.append(0)
    return handled_list

def handleService(input):
    handled_list = []
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u','echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois','X11','Z39_50']
    for i in range(len(service_list)):
        if input[1] == service_list[i]:
            handled_list.append(1)
        else:
            handled_list.append(0)
    return handled_list
def handleFlag(input):
    handled_list = []
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    for i in range(len(flag_list)):
        if input[1] == flag_list[i]:
            handled_list.append(1)
        else:
            handled_list.append(0)
    return handled_list

def handleLabel(input):
    labels_list = [['normal.'],
    ['ipsweep.','mscan.','nmap.','portsweep.','saint.','satan.'],
    ['apache2.','back.','land.','mailbomb.','neptune.','pod.','processtable.','smurf.','teardrop.','udpstorm.'],
    ['buffer_overflow.','httptunnel.','loadmodule.','perl.','ps.','rootkit.','sqlattack.','xterm.'],
    ['ftp_write.','guess_passwd.','imap.','multihop.','named.','phf.','sendmail.','snmpgetattack.','snmpguess.','spy.','warezclient.','warezmaster.','worm.','xlock.','xsnoop.']]

    for i in range(len(labels_list)):
        if input[41] in labels_list[i]:
            return i
    return 0

"""
下面的代码用于将数据处理为2分类
"""
def preHandle_2_class(inputfile, outputfile):
    print(inputfile, outputfile)
    source_file = inputfile
    handled_file = outputfile
    data_to_flie=open(handled_file, 'w')
    with (open(source_file,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        count=0
        for i in csv_reader:
            temp_line=np.array(i)          

            handled_protocol = np.array(handleProtocol(i))  #将源文件行中3种协议类型转换成数字标识
            handled_service = np.array(handleService(i))    #将源文件行中70种网络服务类型转换成数字标识
            handled_flag = np.array(handleFlag(i))  #将源文件行中11种网络连接状态转换成数字标识

            temp_line[41]=handleLabel_2_class(i)          #将源文件行中23种攻击类型转换成数字标识

            temp_line = np.delete(temp_line, [1,2,3])   # 删除这三列，它们已经转为[1,0,0,...,0]的形式了

            temp_line = np.concatenate((temp_line,handled_protocol, handled_service, handled_flag),axis=0)

            csv_writer.writerow(temp_line)
            #print count,'staus:',temp_line[1],temp_line[2],temp_line[3],temp_line[41]
            count+=1
            
        print('processed %d lines data' % count)
        data_to_flie.close()

def handleLabel_2_class(input):
    labels_list = [['normal.'],
    ['ipsweep.','mscan.','nmap.','portsweep.','saint.','satan.'],
    ['apache2.','back.','land.','mailbomb.','neptune.','pod.','processtable.','smurf.','teardrop.','udpstorm.'],
    ['buffer_overflow.','httptunnel.','loadmodule.','perl.','ps.','rootkit.','sqlattack.','xterm.'],
    ['ftp_write.','guess_passwd.','imap.','multihop.','named.','phf.','sendmail.','snmpgetattack.','snmpguess.','spy.','warezclient.','warezmaster.','worm.','xlock.','xsnoop.']]

    if input[41] == 'normal.':
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    # preHandle('/Users/johnson/Downloads/graduation_project/dataset/kddcup.data','kddcup.data_handled.csv')
    # preHandle('/Users/johnson/Downloads/graduation_project/dataset/corrected', 'corrected_handled.csv')
    preHandle_2_class('/Users/johnson/Downloads/graduation_project/NIDS-with-DL/dataset/10_percent_unhandled.csv','./dataset/train_handled_2_class.csv')
    preHandle_2_class('/Users/johnson/Downloads/graduation_project/NIDS-with-DL/dataset/corrected_unhandled.csv', './dataset/test_handled_2_class.csv')