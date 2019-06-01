"""
This script is used to generate the data we need in our experiments.
"""
import urllib.request, os

source_case = ['http://www.med.harvard.edu/AANLIB/cases/case2/', #mr2 ct1
'http://www.med.harvard.edu/AANLIB/cases/case16/', #mr1 ct1
'http://www.med.harvard.edu/AANLIB/cases/case20/', #mr3 ct1
'http://www.med.harvard.edu/AANLIB/cases/case21/', #mr2 ct1
'http://www.med.harvard.edu/AANLIB/cases/case34/', #mr1 ct1
'http://www.med.harvard.edu/AANLIB/cases/case37/', #mr1 ct2
'http://www.med.harvard.edu/AANLIB/cases/case32/', #mr4 ct2
'http://www.med.harvard.edu/AANLIB/cases/case28/', #mr1 ct1
'http://www.med.harvard.edu/AANLIB/cases/case33/', #mr1 ct1
'http://www.med.harvard.edu/AANLIB/cases/case41/'] # mr1 ct1

source_type = [('mr2', 'ct1'), ('mr1', 'ct1'), ('mr3', 'ct1'), ('mr2', 'ct1'), ('mr1', 'ct1'), ('mr1', 'ct2'), ('mr4', 'ct2'), ('mr1', 'ct1'), ('mr1', 'ct1'), ('mr1', 'ct1')]

if not os.path.exists('./data'):
    os.mkdir('./data')


for case in range(1, 11):
    dirs = 'case{}'.format(case)
    if not os.path.exists('./data/' + dirs):
        os.mkdir('./data/' + dirs)
    else:
        print('./data/{} already exists!'.format(dirs))
        
    img_src = source_case[case - 1]
    
    for index, types in enumerate(['mr2', 'ct1']):
        for name in range(0, 24):
            pic_name = str(name).zfill(3) + '.gif'
            print('{}{}/{}'.format(img_src, source_type[case - 1][index], pic_name))
            response = urllib.request.urlopen('{}{}/{}'.format(img_src, source_type[case - 1][index], pic_name))
            cat = response.read()
            with open("./data/" + dirs + "/" + types + "_" + pic_name, "wb") as f:
                f.write(cat)
                print("./data/" + dirs + "/" + types + "_" + pic_name + " save sucess !")
                
