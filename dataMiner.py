import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime
import numpy as np

class DataMiner:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets =[]

    def __init__(self, tiker, batch_size):

        self.__tiker = tiker
        self.__batch_size = batch_size
        #self.__list_tickers()
        self.__read_data_new()
        #self.check_dictionary('/data/output/')

    def __get_tickers(self, subdir):

        return [re.findall(r'.*_(.*)\.\w{3}', f.name)[0] for f in os.scandir(self.__fileDir+ subdir) if f.is_file()]

    def __list_tickers(self):

        dirOutput = '/data/stocks/'
        array = []
        for __ticker in self.__get_tickers(dirOutput):
            array.append(__ticker)

        print(array)

    def __read_data_new(self):

        __ticker = 'Unknown file'
        dirOutput = '/data/stocks/'

        __tickers_array = [ 'SEIC', 'SELF', 'SENEA', 'SENEB', 'SENS', 'SEP', 'SERV', 'SF', 'SFBC',
                           'SFBS', 'SFL', 'SFM', 'SFNC', 'SFUN', 'SGA', 'SGC', 'SGMA', 'SGMO', 'SGOC', 'SGRP', 'SGRY',
                           'SGU', 'SHAK', 'SHE', 'SHEN', 'SHG', 'SHIP', 'SHLM', 'SHLO', 'SHO', 'SHOO', 'SHOP', 'SHPG',
                           'SHSP', 'SHW', 'SID', 'SIEB', 'SIF', 'SIFY', 'SIG', 'SIGI', 'SIGM', 'SILC', 'SIM', 'SIMO',
                           'SINA', 'SINO', 'SIR', 'SIRI', 'SJI', 'SJM', 'SJT', 'SKM', 'SKY', 'SKYS', 'SLAB', 'SLB',
                           'SLCA', 'SLCT', 'SLF', 'SLGN', 'SLM', 'SLMBP', 'SLNO', 'SLP', 'SLRC', 'SM', 'SMBC', 'SMCI',
                           'SMED', 'SMFG', 'SMG', 'SMHD', 'SMIN', 'SMIT', 'SMM', 'SMMF', 'SMMT', 'SMP', 'SMSI', 'SMTC',
                           'SMTS', 'SNA', 'SNBR', 'SNCR', 'SNDX', 'SNE', 'SNFCA', 'SNI', 'SNMP', 'SNMX', 'SNN', 'SNOA',
                           'SNP', 'SNPS', 'SNR', 'SNV', 'SNX', 'SO', 'SODA', 'SOHO', 'SOHU', 'SOJA', 'SONA', 'SOR',
                           'SOVB', 'SPAR', 'SPB', 'SPCB', 'SPDW', 'SPEM', 'SPGI', 'SPH', 'SPHS', 'SPIB', 'SPIL', 'SPKE',
                           'SPLG', 'SPLK', 'SPLP', 'SPMD', 'SPMO', 'SPNE', 'SPNS', 'SPOK', 'SPPI', 'SPPP', 'SPR',
                           'SPRT', 'SPSC', 'SPTM', 'SPTN', 'SPTS', 'SPWR', 'SPXC', 'SPXE', 'SPXN', 'SPXV', 'SPYD', 'SQ',
                           'SQM', 'SQNS', 'SR', 'SRAX', 'SRC', 'SRCL', 'SRDX', 'SRE', 'SREV', 'SRI', 'SRLP', 'SRNE',
                           'SRPT', 'SRT', 'SRV', 'SSB', 'SSBI', 'SSC', 'SSD', 'SSI', 'SSKN', 'SSL', 'SSNC', 'SSNT',
                           'SSP', 'SSRM', 'SSTK', 'SSY', 'SSYS', 'STAA', 'STAG', 'STAR', 'STAY', 'STBA', 'STC', 'STE',
                           'STI', 'STK', 'STKL', 'STKS', 'STLY', 'STM', 'STN', 'STNG', 'STOT', 'STPP', 'STRA', 'STRL',
                           'STRM', 'STRP', 'STRS', 'STT', 'STWD', 'STZ-B', 'STZ', 'SU', 'SUI', 'SUM', 'SUMR', 'SUN',
                           'SUNS', 'SUNW', 'SUP', 'SUPN', 'SUSA', 'SVA', 'SVBI', 'SVT', 'SVU', 'SVVC', 'SWIR', 'SWK',
                           'SWKS', 'SWM', 'SWN', 'SWX', 'SWZ', 'SXC', 'SYBX', 'SYK', 'SYKE', 'SYN', 'SYNA', 'SYNC',
                           'SYNL', 'SYNT', 'SYPR', 'SYX', 'SYY', 'SZC', 'T', 'TAC', 'TACO', 'TAIT', 'TANH', 'TANNL',
                           'TANNZ', 'TARO', 'TAST', 'TAT', 'TATT', 'TAYD', 'TBB', 'TBK', 'TBNK', 'TBPH', 'TCBI',
                           'TCBIL', 'TCBK', 'TCCO', 'TCF', 'TCFC', 'TCI', 'TCO', 'TCPC', 'TCS', 'TCX', 'TD', 'TDA',
                           'TDC', 'TDE', 'TDF', 'TDI', 'TDJ', 'TDOC', 'TDW', 'TEAM', 'TECD', 'TECH', 'TECK', 'TEDU',
                           'TELL', 'TEN', 'TENX', 'TEP', 'TER', 'TERP', 'TESS', 'TEUM', 'TEVA', 'TFX', 'TGA', 'TGB',
                           'TGC', 'TGEN', 'TGH', 'TGI', 'TGLS', 'TGNA', 'TGP', 'TGS', 'TGT', 'TGTX', 'THC', 'THG',
                           'THGA', 'THM', 'THO', 'THRM', 'THS', 'THST', 'THW', 'TIF', 'TIPT', 'TISI', 'TITN', 'TIVO',
                           'TJX', 'TK', 'TKAT', 'TKC', 'TKR', 'TLDH', 'TLEH', 'TLF', 'TLGT', 'TLI', 'TLK', 'TLRA',
                           'TLRD', 'TLYS', 'TM', 'TMP', 'TMQ', 'TMST', 'TMUS', 'TNAV', 'TNDM', 'TNET', 'TNH', 'TNP',
                           'TNXP', 'TOO', 'TOPS', 'TOUR', 'TPC', 'TPH', 'TPHS', 'TPL', 'TPR', 'TPVG', 'TPYP', 'TPZ',
                           'TR', 'TRC', 'TREC', 'TREE', 'TREX', 'TRGP', 'TRI', 'TRIB', 'TRIP', 'TRMB', 'TRMK', 'TRN',
                           'TRNO', 'TRNS', 'TROV', 'TROW', 'TROX', 'TRP', 'TRQ', 'TRS', 'TRST', 'TRT', 'TRTN', 'TRU',
                           'TRUE', 'TRUP', 'TRV', 'TRVN', 'TRX', 'TRXC', 'TS', 'TSBK', 'TSC', 'TSCO', 'TSE', 'TSEM',
                           'TSG', 'TSI', 'TSLA', 'TSLX', 'TSM', 'TSN', 'TSRO', 'TSU', 'TTC', 'TTEC', 'TTEK', 'TTGT',
                           'TTI', 'TTM', 'TTMI', 'TTP', 'TTWO', 'TU', 'TUES', 'TUP', 'TURN', 'TV', 'TVC', 'TVE', 'TVPT',
                           'TVTY', 'TWI', 'TWLO', 'TWMC', 'TWN', 'TWNK', 'TWO', 'TWOU', 'TWTR', 'TX', 'TXMD', 'TXN',
                           'TY', 'TYG', 'TYL', 'TYME', 'TZOO', 'UA', 'UAA', 'UAL', 'UAN', 'UBA', 'UBCP', 'UBIO', 'UBOH',
                           'UBS', 'UBSI', 'UCBA', 'UCBI', 'UCFC', 'UEC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS',
                           'UG', 'UGI', 'UGP', 'UHAL', 'UHS', 'UHT', 'UL', 'ULTA', 'UMC', 'UMH', 'UMPQ', 'UN', 'UNB',
                           'UNM', 'UNP', 'UNT', 'UNTY', 'UNVR', 'UONEK', 'UPL', 'UPS', 'URBN', 'URG', 'URI', 'USA',
                           'USAC', 'USAK', 'USAP', 'USAS', 'USAT', 'USATP', 'USB', 'USCR', 'USEG', 'USLB', 'USLM',
                           'USM', 'USNA', 'USRT', 'UTES', 'UTF', 'UTG', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTSI', 'UUU',
                           'UUUU', 'UVSP', 'UVV', 'UZA', 'UZB', 'V', 'VAC', 'VALE', 'VALU', 'VALX', 'VAR', 'VBF',
                           'VBFC', 'VBIV', 'VBLT', 'VBND', 'VC', 'VCEL', 'VCF', 'VCYT', 'VEC', 'VECO', 'VEDL', 'VEEV',
                           'VEON', 'VER', 'VERU', 'VET', 'VFC', 'VFL', 'VG', 'VGI', 'VGM', 'VGZ', 'VHC', 'VHI', 'VIAV',
                           'VICR', 'VIPS', 'VIRT', 'VIV', 'VIVE', 'VIVO', 'VJET', 'VKI', 'VKQ', 'VKTX', 'VLGEA', 'VLO',
                           'VLP', 'VLRS', 'VLY', 'VMC', 'VMI', 'VMO', 'VMW', 'VNCE', 'VNET', 'VNO', 'VNOM', 'VOC',
                           'VOXX', 'VR', 'VRA', 'VRML', 'VRNS', 'VRNT', 'VRSK', 'VRSN', 'VRTS', 'VRTU', 'VRTV', 'VRTX',
                           'VSEC', 'VSH', 'VSLR', 'VST', 'VSTM', 'VSTO', 'VTA', 'VTEB', 'VTN', 'VTNR', 'VTR', 'VTVT',
                           'VVI', 'VYGR', 'VYMI', 'VZA', 'W', 'WAB', 'WAFD', 'WAL', 'WASH', 'WAT', 'WATT', 'WBA',
                           'WBBW', 'WBIA', 'WBIE', 'WBIF', 'WBIG', 'WBIH', 'WBII', 'WBIL', 'WBK', 'WBS', 'WBT', 'WCC',
                           'WCFB', 'WCN', 'WDAY', 'WDC', 'WDFC', 'WDR', 'WEA', 'WEB', 'WEBK', 'WEN', 'WERN', 'WETF',
                           'WEX', 'WFC', 'WGL', 'WGO', 'WHF', 'WHG', 'WHLM', 'WHLR', 'WHR', 'WIA', 'WIFI', 'WINA',
                           'WING', 'WINS', 'WIT', 'WIW', 'WKHS', 'WLDN', 'WLFC', 'WLL', 'WMC', 'WMGI', 'WMS', 'WMT',
                           'WNC', 'WNEB', 'WNS', 'WOR', 'WPC', 'WPG', 'WPRT', 'WPX', 'WPZ', 'WRE', 'WRI', 'WRK', 'WRLD',
                           'WSBC', 'WSCI', 'WSM', 'WSO-B', 'WSO', 'WSR', 'WST', 'WTBA', 'WTFC', 'WTFCM', 'WTI', 'WTM',
                           'WTR', 'WTS', 'WUBA', 'WVE', 'WVFC', 'WVVI', 'WWD', 'WWE', 'WWR', 'WWW', 'WY', 'WYY', 'X',
                           'XBIT', 'XCEM', 'XCRA', 'XEC', 'XEL', 'XELB', 'XENT', 'XHR', 'XIN', 'XITK', 'XL', 'XLNX',
                           'XLRE', 'XLRN', 'XNCR', 'XNET', 'XNTK', 'XOM', 'XOMA', 'XONE', 'XOXO', 'XPER', 'XPL', 'XPO',
                           'XRAY', 'XRM', 'XTLB', 'XXII', 'XYL', 'Y', 'YELP', 'YLD', 'YNDX', 'YORW', 'YRCW', 'YRD',
                           'YTEN', 'YUM', 'YY', 'Z', 'ZAIS', 'ZAYO', 'ZBH', 'ZBIO', 'ZBK', 'ZBRA', 'ZEN', 'ZGNX',
                           'ZION', 'ZIOP', 'ZIXI', 'ZN', 'ZNH', 'ZOES', 'ZSAN', 'ZTR', 'ZTS', 'ZYNE']

        try:

            #for __ticker in self.__get_tickers(dirOutput):
            for __ticker in __tickers_array:

                raw_data = []
                with open(self.__fileDir + dirOutput + 'train_' + __ticker + '.csv', newline='') as f:

                    next(f)
                    rows = csv.reader(f, delimiter=',', quotechar='|')
                    row = next(rows)
                    print("---- " + __ticker + " ----------")

                    while True:
                        try:

                            next_row = next(rows)

                            # Find carrier as a change of open in percentages
                            carrier = self.__change_percent(row[1], next_row[1])
                            row = next_row

                            # Find day of year
                            day_of_year = self.__day_of_year(row[0])
                            row.append(day_of_year)
                            row.append(carrier)

                            low_ = self.__change_percent(row[1], row[2])
                            row.append(low_)

                            high_ = self.__change_percent(row[1], row[3])
                            row.append(high_)

                            close_current_ = self.__change_percent(row[1], row[4])
                            row.append(close_current_)

                            raw_data.append(row)

                        except StopIteration:
                            break

                f.close()

                X_array_ticker, Y_array_tiker = self.__calculate_col_values(3, raw_data)
                self.__append_to_file(__ticker, raw_data)

        except FileNotFoundError:

            print('Error: File "' + __ticker + '.csv" not found')

    # Check dictionary
    def check_dictionary(self, dirname):

        __ticker = 'Unknown file'
        __d = {}
        count = 0

        print("-------- Dictionary check  ---->> " )
        try:
            for __ticker in self.__get_tickers(dirname):

                raw_data = []
                with open(self.__fileDir + dirname + 'train_' + __ticker + '.csv', newline='') as f:

                    print("---->> " + str(__ticker))

                    next(f)
                    rows = csv.reader(f, delimiter=';', quotechar='|')

                    for row in rows:
                        raw_data.append(row)

                    X_array, Y_array = self.__calculate_col_values(3, raw_data)

                    count += X_array.shape[0]
                    for i in range (0, X_array.shape[0]):

                        key = ";".join(np.array(X_array[i]).flatten().astype(str))

                        if key in __d:
                            __d[key] = __d[key] + 1
                        else:
                            __d[key] = 1
                    print( "Current count :"+str(count))

                f.close()

            print("Common rows : " + str(count) + " dictionary size : " + str(len(__d)))

            i = 0
            if (len(__d) < count):

                for key, value in sorted(__d.items(), key=lambda x: x[1]):

                    if (value > 1):
                        i +=1
                        print("Key :" + str(key) + ", value :" + str(value))

            print("There is dubled " + str(i))

        except FileNotFoundError:
            print('Error: File "' + __ticker + '.csv" not found')


    # Write data to output csv file
    def __append_to_file(self, name, data):

        filename = self.__fileDir + '/data/rawdata/train_' + name + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')

            output.writerow(['Date', 'Open', 'Low', 'High', 'Close', 'Adj' 'Close', 'Volume',
                            'DAY', 'Carrier', "Low'", "High'", "Close/Current'"])

            for line in data:
                output.writerow(line)

        csv_out_file.close()


    def __change_percent(self, current, next):

        return float(D((float(next) - float(current))/float(current)*100).quantize(D('0.1'), rounding=ROUND_DOWN))

    def __calculate_col_values(self, range_size, raw_data):

        n_array = (np.delete(np.array(raw_data), (0, 1, 5, 6, 7), axis=1)).astype(np.float64)
        data_len = n_array.shape[0]

        Y_array = []
        X_array = []

        for i in range(0, data_len - 2 * range_size + 1):
            end = i + range_size

            # Find max & min in feature slice period
            f_max = np.max(n_array[i + range_size:end + range_size], axis=0)[1]
            f_min = np.min(n_array[i + range_size:end + range_size], axis=0)[0]


            # Find Low & High change in feature slice period
            f_ch_percent_low = self.__change_percent(str(n_array[i:end][-1][2]), f_min)
            f_ch_percent_high = self.__change_percent(str(n_array[i:end][-1][2]), f_max)

            # Remove abs values from array
            X_row = np.delete(n_array[i:end], np.s_[0, 1, 2], 1)
            Y_row = np.array([f_ch_percent_low,f_ch_percent_high])

            Y_array.append(Y_row)
            X_array.append(X_row)

        return np.array(X_array), np.array(Y_array)


    # Convert date to day of year
    def __day_of_year(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date().timetuple().tm_yday
