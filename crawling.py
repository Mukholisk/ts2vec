job_number = input()
infile = open('./slurm/logs/mukho-' + job_number + '-train_test.out', 'r')
outfile = open('./slurm/result/' + job_number + '-result.out', 'w')

dataset_type = ''
temp = []
check = False

UCR_count = 0
UEA_count = 0
UCR_acc = 0
UEA_acc = 0
UCR_loss = 0
UEA_loss = 0
for content in infile:
    if content.startswith('Arguments:'):
        temp.clear()
        if content.find('UCR') != -1:
            dataset_type = 'UCR'
        else:
            dataset_type = 'UEA'
    if content.startswith('Epoch #'):
        check = True
        temp.append(content)
    if check == True and content == '\n':
        front_idx = temp[-1].index('=')+1
        if dataset_type == 'UCR':
            UCR_count += 1
            UCR_loss += float(temp[-1][front_idx:])
        else:
            UEA_count += 1
            UEA_loss += float(temp[-1][front_idx:])
        check = False
        temp.clear()
    if content.startswith('Evaluation result: '):
        front_idx = content.index("'acc': ")+7
        back_idx = content.index(', ')
        print(dataset_type, float(content[front_idx:back_idx]))
        if dataset_type == 'UCR':
            UCR_acc += float(content[front_idx:back_idx])
        else:
            UEA_acc += float(content[front_idx:back_idx])
        dataset_type = ''
    
infile.close()

outfile.write('count / acc avg. / loss avg.\n')
outfile.write('dataset: UCR\n')
outfile.write(str(UCR_count) + '/' + str(UCR_acc/UCR_count) + '/' + str(UCR_loss/UCR_count) + '\n')
outfile.write('dataset: UEA\n')
outfile.write(str(UEA_count) + '/' + str(UEA_acc/UEA_count) + '/' + str(UEA_loss/UEA_count))
outfile.close()
