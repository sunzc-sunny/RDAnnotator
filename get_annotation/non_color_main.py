from get_annotation.rdannotator import AnnTool  

if __name__ == '__main__':

    image_dir = "/home/sunzc/VisDrone2019/sample_visdrone_night_image_sample"
    caption_save_dir = "/home/sunzc/VisDroneAnnotation/color_caption_sample"
    color_check_save_dir = "/home/sunzc/VisDroneAnnotation/night_color_check_sample"
    color_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/night_color_annotation_sample"
    noncolor_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/night_noncolor_annotation_sample"
    color_check_annotation_save_dir = "/home/sunzc/VisDroneAnnotation/night_color_check_annotation_sample"
    noncolor_check_annotation_save_dir = "/home/sunzc/VisDroneAnnotation/night_noncolor_check_annotation_sample"
    color_regenerate_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/night_color_regenerate_annotation_sample"
    noncolor_regenerate_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/night_noncolor_regenerate_annotation_sample"

    anntool = AnnTool(image_dir, caption_save_dir=caption_save_dir, color_check_save_dir=color_check_save_dir, color_annotator_save_dir=color_annotator_save_dir, 
    noncolor_annotator_save_dir=noncolor_annotator_save_dir, color_check_annotation_save_dir=color_check_annotation_save_dir, noncolor_check_annotation_save_dir=noncolor_check_annotation_save_dir,
    color_regenerate_annotator_save_dir=color_regenerate_annotator_save_dir, noncolor_regenerate_annotator_save_dir=noncolor_regenerate_annotator_save_dir)
    anntool.noncolor_run(image_dir)


