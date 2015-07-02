def main():
	BproComparision = open('BproJobs.txt','a')
	for i in range(1,31):
		line = 'Bpro-Pong-Trial'+str(i)+'\t./learnerBpro -s '+str(i)+' -c bpro.cfg -r ../roms/pong.bin -n Bpro-Pong-Trial'+str(i)+'\n'
		BproComparision.write(line)
	BproComparision.close()


main()