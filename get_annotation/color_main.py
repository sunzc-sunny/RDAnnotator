from get_annotation.rdannotator import AnnTool

if __name__ == '__main__':

    image_dir = "/home/sunzc/VisDrone2019/test_color_image_sample"
    color_info_dir = "/home/sunzc/VisDrone2019/test_color_image_sample_ann"

    caption_save_dir = "/home/sunzc/VisDroneAnnotation/test_color_caption_sample"
    color_check_save_dir = "/home/sunzc/VisDroneAnnotation/test_color_check_sample"
    color_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/test_color_annotation_sample"
    noncolor_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/test_noncolor_annotation_sample"
    color_check_annotation_save_dir = "/home/sunzc/VisDroneAnnotation/test_color_check_annotation_sample"
    noncolor_check_annotation_save_dir = "/home/sunzc/VisDroneAnnotation/test_noncolor_check_annotation_sample"
    color_regenerate_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/test_color_regenerate_annotation_sample"
    noncolor_regenerate_annotator_save_dir = "/home/sunzc/VisDroneAnnotation/test_noncolor_regenerate_annotation_sample"



    anntool = AnnTool(image_dir, color_info_dir=color_info_dir, caption_save_dir=caption_save_dir, color_check_save_dir=color_check_save_dir, color_annotator_save_dir=color_annotator_save_dir, 
    noncolor_annotator_save_dir=noncolor_annotator_save_dir, color_check_annotation_save_dir=color_check_annotation_save_dir, noncolor_check_annotation_save_dir=noncolor_check_annotation_save_dir,
    color_regenerate_annotator_save_dir=color_regenerate_annotator_save_dir, noncolor_regenerate_annotator_save_dir=noncolor_regenerate_annotator_save_dir)
    anntool.color_run(image_dir)


