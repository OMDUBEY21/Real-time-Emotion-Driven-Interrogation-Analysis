from emotion_reports import PDF_Report
from bson import ObjectId
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
collection = client.EmDetect.reports
report_name = ''
suspect_name = ''
# oid = '644b6fbf59c8f15e7f7c3e68'

def generate_pdf(oid):
    pdf = PDF_Report()
    pdf.alias_nb_pages()
    report = collection.find_one({'_id': ObjectId(oid)})
    report_name = report['case_name']
    suspect_name = report['suspect_name']
    print('**********GENERATING REPORT***********')
    for session_num, session in enumerate(report['sessions'], start=1):
        print('Generating Session ', session_num)

        pdf.add_page() 

        pdf.set_session(session_num, report_name, suspect_name, oid)
        pdf.ln()

        # pdf.image(pdf.plot_tl_line_chart(), w=pdf.epw)
        pdf.image(pdf.plot_tl_pie_chart([session['truth_percent'], session['lie_percent'], session['neutral_percent']]), w=300, x=2, y=70)

        blink_val_list = [val for val in session['blink_stat'].values()]
        pdf.set_blink_rate_stats(blink_val_list)
        pdf.ln()

        em_val_list = [val for val in session['emotion_stat'].values()]
        pdf.image(pdf.plot_em_bar_graph(em_val_list), w=pdf.epw, x=12, y=190)
    pdf.output(report_name+'.pdf')
    
    print('*****************DONE*****************')

# generate_pdf('644b6fbf59c8f15e7f7c3e68')
