def main():
	BproComparision = open('BproComparision.txt','a')
	for i in range(1,31):
		line = 'Redudant-Asterix-Trial'+str(i)+'\t./learnerRedundant -s '+str(i)+' -c /home/yliang/Research/conf/bpro/cfg -r /home/yliang/Research/roms/asterix.bin -w Redundant-Asterix-Trial'+str(i)+'\n'
		BproComparision.write(line)
	BproComparision.close()


main()