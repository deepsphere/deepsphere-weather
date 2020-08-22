import smtplib

import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_info_mail(mailInfo, configFile):

    fromaddr = mailInfo['sender']
    toaddr = mailInfo['receiver']

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = mailInfo['subject']

    body = mailInfo['body']

    msg.attach(MIMEText(body, 'plain'))

    if mailInfo['fileAttaching']:
        for file_to_attach in mailInfo['file']:
            filename = file_to_attach
            attachment = open(filename, "rb")

            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, configFile['password'])
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

