from marker.scripts.convert_single import convert_single_cli
import sys
import os 

sys.argv = ['marker_single', 
            "/home/lxy/case_engine/ChinaVis_2025___JOV___TemporalFlowViz__A_Visual_Analytics_Approach_to_Analyse_Temporal_Flow_Field_of_Scramjet_Combustion.pdf",
            '--output_format', 'markdown',
            '--output_dir', '/home/lxy/case_engine',]
try:
    convert_single_cli()
except Exception as e:
    import traceback
    traceback.print_exc()
