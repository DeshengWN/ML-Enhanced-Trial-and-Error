import re

#operators = ['+', '-', '*', '/', '(', ')']
operators = ['sub', 'sqrt', 'div', 'add', 'mul']
file1=open(f"./1.1SR/300generation/count/y4_Elon.txt",mode="a")

with open('./1.1SR/300generation/yin-Equation_5_Elon.txt') as f:
    for line in f:
        nums = line.split() 
        idx = int(nums[0])
        
        formula = next(f).strip()
        

        count = 0

        for op in operators:
          count += formula.count(op)
        r21 = float(nums[1])
        r22 = float(nums[2])
        #writer.writerow([idx, count, r2])  
    
        print(idx,round(r21,3),round(r22,3),count,file=file1)
            
        print(f"For index {idx}, operator count is {count}")