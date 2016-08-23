import pandas as pd
import matplotlib.pyplot as plt



def clean_top(df_original):
    df = df_original.copy()
    '''
    array([u'acct_type', u'approx_payout_date', u'body_length', u'channels', u'country', u'currency', u'delivery_method', u'description',
       u'email_domain', u'event_created', u'event_end', u'event_published',
       u'event_start', u'fb_published', u'gts', u'has_analytics',
       u'has_header', u'has_logo', u'listed', u'name', u'name_length',
       u'num_order', u'num_payouts', u'object_id', u'org_desc',
       u'org_facebook', u'org_name', u'org_twitter', u'payee_name',
       u'payout_type', u'previous_payouts', u'sale_duration',
       u'sale_duration2', u'show_map', u'ticket_types', u'user_age',
       u'user_created', u'user_type', u'venue_address', u'venue_country',
       u'venue_latitude', u'venue_longitude', u'venue_name',
       u'venue_state', 'fraud'], dtype=object)
    '''

    common_domains = [u'GMAIL.COM', u'YAHOO.COM', u'HOTMAIL.COM', u'AOL.COM', u'LIVE.COM',
       u'ME.COM', u'YMAIL.COM', u'COMCAST.NET', u'GENERALASSEMB.LY',
       u'YAHOO.CO.UK', u'KINETICEVENTS.COM', u'HOTMAIL.CO.UK',
       u'IMPROVBOSTON.COM', u'SIPPINGNPAINTING.COM', u'LIVE.FR',
       u'CLAYTONISLANDTOURS.COM', u'RACETONOWHERE.COM', u'LIDF.CO.UK',
       u'YAHOO.CA', u'GREATWORLDADVENTURES.COM', u'SHAW.CA',
       u'SBCGLOBAL.NET', u'MAC.COM', u'MSN.COM', u'GUARDIAN.CO.UK',
       u'LIVE.CO.UK', u'JOONBUG.COM', u'ROCKETMAIL.COM', u'JHILBURN.COM',
       u'YAHOO.FR', u'BUSBOYSANDPOETS.COM', u'DSICOMEDY.COM',
       u'SENECALAKEWINE.COM', u'O-CINEMA.ORG', u'DOCTOR.CNC.NET',
       u'THEMAGNETICTHEATRE.ORG', u'WOMENLIKEUS.ORG.UK', u'VERIZON.NET',
       u'WHOLEFOODS.COM', u'WEBOOKBANDS.COM', u'GOOGLEMAIL.COM',
       u'COX.NET', u'OUTLOOK.COM', u'DISCODONNIEPRESENTS.COM',
       u'JUMPNASIUMPARTY.COM', u'TRIBECAFILMFESTIVAL.ORG',
       u'PRICECUTTERONLINE.COM', u'ATT.NET', u'AMERICANPHOTOSAFARI.COM',
       u'HIGHGATE-CEMETERY.ORG', u'DULWICHPICTUREGALLERY.ORG.UK',
       u'COMIXENTERTAINMENT.COM', u'BIGCITYMOMS.COM',
       u'ULTIMATEWINE.CO.UK', u'GREENEDU.COM', u'FIRSTAM.COM',
       u'ABELCINE.COM', u'FSB.ORG.UK', u'GIANTEAGLE.COM',
       u'EDGEQLD.ORG.AU', u'CAMAJE.COM', u'BSOP.CA', u'AUSTMUS.GOV.AU',
       u'SAMYS.COM', u'KW.COM', u'HAM.HONDA.COM', u'BMCJAX.COM', u'.COM',
       u'GER-NIS.COM', u'STARTUPGRIND.COM', u'FABRICATIONS1.CO.UK',
       u'YOPMAIL.COM', u'MENTA.ORG.UK', u'LEADERIMPACTGROUP.COM',
       u'ROGERS.COM', u'MAIL.COM', u'MONGODB.COM', u'FACIALESTHETICS.ORG',
       u'ABFABPARTIES.COM', u'DIVERSITY-CHURCH.COM', u'OPTONLINE.NET',
       u'EARTHLINK.NET', u'UFL.EDU', u'BSO.AC.UK', u'PASSIONALTOYS.COM',
       u'KAYAKINGLONDON.COM', u'SDCPLL.ORG', u'BUSTOSHOW.ORG',
       u'OPENCITYLONDON.COM', u'TELUS.NET', u'NYU.EDU',
       u'CRAFTOFCOCKTAILS.COM', u'KIDSANDCOMPANY.CA',
       u'BLACKJACKETGROUP.NET', u'GEORGETOWN.EDU',
       u'KINGSTONCHAMBER.CO.UK', u'JAZZSCHOOL.ORG', u'FORESTCITY.NET',
       u'EXECULINK.COM', u'SITE3.CA'],


    df['approx_payout_date'] = pd.to_datetime(df['approx_payout_date'],unit='s')
    # most common countries = 1, else 0
    df['country'] = df['country'].apply(lambda x: 1 if x in ('US','GB','CA','AU','NZ') else 0)
    df['currency'] = df['currency'].apply(lambda x: 1 if x=='USD' else 2 if x=='EUR' else 3 if x=='CAD' else 4 if x=='GBP' else 5 if x=='AUD' else 6 if x=='NZD' else 7 if x=='MXN' else 8)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 5 if pd.isnull(x) else x).astype(int)
    df['description_char_length'] = df['description'].apply(lambda x: len(x))
    df['description_caps_pct'] =  df['description'].apply(lambda x: (sum(1 for y in x if y.isupper())+1)/float(len(x)+1))
    df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))
    df['email_domain'] = df['email_domain'].apply(lambda x: x.upper())
    
    df['common_email_domain'] = df['email_domain'].apply(lambda x: 1 if x in common_domains else 0)
    df['event_created'] = pd.to_datetime(df['event_created'],unit='s')
    df['event_end'] = pd.to_datetime(df['event_end'],unit='s')
    df['event_published'] = pd.to_datetime(df['event_published'],unit='s')
    df['event_published'][pd.isnull(df['event_published'])] = df['event_created'][pd.isnull(df['event_published'])]
    df['event_start'] = pd.to_datetime(df['event_start'],unit='s')
    # gts = gross total sales
    df['has_header'] = df['has_header'].apply(lambda x: 0 if pd.isnull(x) else x).astype(int)
    df['listed'] = df['listed'].apply(lambda x: 1 if x=='y' else 0)
    df['name_caps_pct'] =  df['name'].apply(lambda x: (sum(1 for y in x if y.isupper())+1)/float(len(x)+1))
    # objectID likely unnecessary
    # columns that need NLP techniques: description, name, org_desc
    df['org_facebook'] = df['org_facebook'].apply(lambda x: 100 if pd.isnull(x) else int(x))
    df['org_twitter'] = df['org_twitter'].apply(lambda x: 100 if pd. isnull(x) else x).astype(int)
    df['has_payee_name'] = df['payee_name'].apply(lambda x: 1 if x else 0)
    df['payout_type'] = df['payout_type'].apply(lambda x: 1 if x=='' else 2 if x=='CHECK' else 3 if x=='ACH' else 4)
    # previous_payouts contains address,amount,country,created,event,name,state,uid,zip_code
    df['num_prev_payouts'] = df['previous_payouts'].apply(lambda x: len(x))
    df['sum_prev_payouts'] = df['previous_payouts'].apply(lambda x:sum([payout['amount'] for payout in x]))
    #sale_duration contains 155 nulls
    df['sale_duration'] = df['sale_duration'].apply(lambda x: 0 if pd.isnull(x) else x).astype(int)
    #sale duration2 mostly matches up with sale_duration and doesnt have nulls, get null values from there instead of setting to 0


    return df












if __name__=='__main__':
    df = pd.read_json('../data/train_new.json')
    df['fraud'] = df['acct_type'].apply(lambda x: 1 if x in ['fraudster_event','fraudster','fraudster_att'] else 0)

    cleaned_df = clean_top(df)
