from fpdf import FPDF 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

class PDF_Report(FPDF):

    # Page Header
    def header(self):
        # self.image('one piece.jpg', 10, 8, 15, 15)
        self.set_font('Helvetica', 'B', 20)
        # self.cell(80)
        self.cell(self.epw, 10, 'EmDetect', 0, 0, 'C')
        self.line(x1 = 10, y1 = 27.5, x2 = 200, y2 = 27.5)
        self.ln(20)

    # Page footer
    def footer(self):
        self.line(x1 = 10, y1 = 276, x2 = 200, y2 = 276)
        self.set_y(-15)
        self.set_font('Helvetica', 'IB', 12)
        self.cell(0, 10, str(self.page_no()), 0, 0, 'R')
        self.set_x(10)
        self.cell(0, 10, '© 2023 EmDetect', 0, 0, 'L')

    # Set session number
    def set_session(self, num, report_name, suspect_name, id):
        self.set_y(30)
        # self.set_x(15)
        self.set_font('Helvetica', 'IB', 18)
        self.set_text_color(228, 96, 0)
        self.cell(w=self.epw, h=5, txt='Session No: ' + str(num), align='L')
        self.ln()
        self.ln()
        self.set_text_color(128, 128, 128)
        self.set_font('Helvetica', 'IB', 14)
        self.cell(w=self.epw, h=5, txt='Case: ' + str(report_name), align='L')
        self.ln()
        self.ln()
        self.set_font('Helvetica', 'IB', 14)
        self.cell(w=self.epw, h=5, txt='Suspect: ' + str(suspect_name), align='L')
        self.ln()
        self.ln()
        self.set_font('Helvetica', 'IB', 14)
        self.cell(w=self.epw, h=5, txt='Id: ' + str(id), align='L')
        self.ln()
        self.ln()
        # self.set_x(5)
        self.set_text_color(128, 128, 128)
        # self.cell(w=self.epw, h=5, txt=id, align='R')

    # Turth-Lie line chart
    def plot_tl_line_chart(self):
        fig = Figure(figsize=(6, 2.5), dpi=300)
        # fig.subplots_adjust(top=0.8)
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel("")
        ax1.set_title("Truth/Lie")

        t = np.arange(0.0, 1.0, 0.01)
        s = np.cos(2 * np.pi * t)
        (line,) = ax1.plot(t, s, color="blue", lw=2)

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        # Converting Figure to an image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        line_chart = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        return line_chart 
    
    # Truth-Lie pie chart
    def plot_tl_pie_chart(self, arr):
        label_val = ['Truth', 'Lie', 'N/A']
        color_val = ['#0CB51C', '#D32323', '#A7A7A7']
        # labels = [label if val != 0 else '' for val, label in zip(arr, label_val)]

        fig = Figure(dpi=300)
        ax = fig.add_subplot(211)
        
        ax.pie(arr, labels=label_val, colors=color_val, textprops={'fontsize': 8}, explode=[0.05, 0.05, 0.05], autopct='%1.1f%%', pctdistance=.75)
        centre_circle = plt.Circle((0, 0), 0.55, fc='white')
        fig.gca().add_artist(centre_circle)

        canvas = FigureCanvas(fig)
        canvas.draw()
        pie_chart = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        return pie_chart 
    
    # Emotions bar graph
    def plot_em_bar_graph(self, arr):
        emotions = ['Anger', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprised']

        fig = Figure(dpi=300)
        ax = fig.add_subplot(211)        
        bars = ax.bar(emotions, arr, color='#F86A04')
        ax.set_ylabel('Occurence')
        ax.set_title('Emotion Stats')
        ax.bar_label(bars, labels=arr, label_type='edge', padding=-15, clip_on=True, color='white', fontweight='bold')

        canvas = FigureCanvas(fig)
        canvas.draw()
        bar_graph = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        return bar_graph 

    # Blink rate stats
    def set_blink_rate_stats(self, arr):
        self.set_y(108)
        blink_rate = [
            ['Blink Rate', 'Value'],
            ['Base', str(arr[0])],
            ['Average', str(arr[1])],
            ['Maximum', str(arr[2])],
        ]  
        self.set_font('Helvetica', 'B', 14)
        for data_row in blink_rate:
            self.set_x(25)
            for datum in data_row:
                if (datum == 'Average'):
                    self.set_text_color(25, 170, 0)
                elif (datum == 'Maximum'):
                    self.set_text_color(235, 0, 0)
                else:
                    self.set_text_color(71, 71, 71)
                self.cell(40, 3.5 * self.font_size, datum, border=1, align='C')
            self.ln()

    # Emotion stats
    def set_emotion_stats(self, surprised, fear, angry, disgust, neutral, happy, sad):
        emotions = [
            ['Emotions', 'Occurence(%)'],
            ['Anger', str(angry)],
            ['Disgust', str(disgust)],
            ['Fear', str(fear)],
            ['Happy', str(happy)],
            ['Neutral', str(neutral)],
            ['Sad', str(sad)],
            ['Surprise', str(surprised)]
        ]  
        self.set_font('Helvetica', 'B', 14)
        for data_row in emotions:
            for datum in data_row:
                if (datum == 'Average'):
                    self.set_text_color(25, 170, 0)
                elif (datum == 'Maximum'):
                    self.set_text_color(235, 0, 0)
                else:
                    self.set_text_color(71, 71, 71)
                self.cell(50, 2.5 * self.font_size, datum, border=1, align='C')
            self.ln()
            

    