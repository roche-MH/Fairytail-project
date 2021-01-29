#!python3
import xml.etree.ElementTree as ET
img_width=1920
img_height=1080
root_path = /home/superbin/??????????
value = "have your class list "
def math_norm(xmin,xmax,ymin,ymax):
	width = xmax-xmin
	height = ymax - ymin
	x_center = xmin + ((xmax-xmin)/2.0)
	y_center = ymin + ((ymax - ymin)/2.0)
	w_norm = width * (1./img_width)
	h_norm = height * (1./img_height)
	x_norm = x_center * (1./img_width)
	y_norm = y_center * (1./img_height)
	return x_norm, y_norm, w_norm, h_norm

def xml2txt(df,save_path):
	for file in df:
		xml = file.split('/')[-1][:-3] + 'xml'
		folder = xml[:7]
		parsedXML = ET.parse(root_path +folde+'/'+xml)
		for node in parsedXML.getroot().iter('object'):
			yolo_list = []
			classes = node.find('name').text
			xmin = int(node.find('bndbox/xmin').text)
			xmax = int(node.find('bndbox/xmax').text)
			ymin = int(node.find('bndbox/ymin').text)
			ymax = int(node.find('bndbox/ymax').text)
			xmin,xmax,ymin,ymax = math_norm(xmin,xmax,ymin,ymax)
			yolo_list.append([value.index(classes),xmin,xmax,ymin,ymax])
		yolo_list = np.array(yolo_list)
		txt_filename = os.path.join(save_path,str(xml[:3]+'txt'))
		np.savetxt(txt_filename,yolo_list,fmt=["%d","%f","%f","%f","%f"])
			