#encoding: utf-8
require 'set'
def conll_prec gold, tst, labeled = false
	labeled = true if labeled != false
	cc = 0
	tc = 0
	gc = 0
	wc = 0
	sc = 0
	f2 = File.open(tst, mode: 'r:utf-8')
	File.open(gold, mode: 'r:utf-8') do |f1|
		over = false
		i = 0
		j = 0
		sent1, sent2 = [], []
		heads1, heads2 = [], []
		heads_inverse1, heads_inverse2 = Hash.new, Hash.new
		f1.each_line do |line1|
			next if line1.start_with?("#")
			line1 = line1.rstrip
			if line1.size == 0
				sc += 1
				arcs1, arcs2 = Set.new, Set.new
				sent1.each_with_index do |token, i|
					next if token == nil
					token.each_with_index do |label, index|
						unless label.eql?("_")
							arcs1 << (!labeled ? [heads1[index], i] : [heads1[index], i, label])
						end
					end
				end
				sent2.each_with_index do |token, i|
					next if token == nil
					token.each_with_index do |label, index|
						unless label.eql?("_")
							arcs2 << (!labeled ? [heads2[index], i] : [heads2[index], i, label])
						end
					end
				end
				arcs21 = arcs2 - arcs1
				wc += 1 if arcs21.size == 0 && arcs2.size == arcs1.size
				cc += (arcs2.size - arcs21.size)
				tc += arcs2.size
				gc += arcs1.size
				sent1, sent2 = [], []
				heads1, heads2 = [], []
				heads_inverse1, heads_inverse2 = Hash.new, Hash.new
				i += 1
				j = 0
			end
			line2 = f2.gets
			break if line2 == nil
			line2 = line2.rstrip
			next if line2.size == 0
			line2 = f2.gets.rstrip if line2.start_with?("#")
			args1 = line1.split(/\s+/)
			args2 = line2.split(/\s+/)
			unless args1[10].eql?("_")
				heads_inverse1[j] = heads1.size
				heads1 << j
			end
			unless args2[10].eql?("_")
				heads_inverse2[j] = heads2.size
				heads2 << j
			end
			sent1 << args1[11..-1]
			sent2 << args2[11..-1]
			j += 1
		end
	end
	f2.close
	p = cc.to_f / tc.to_f
	r = cc.to_f / gc.to_f
	puts "#{p} #{r} #{(2 * p * r) / (p + r)}"
	#puts "#{wc.to_f / sc.to_f}"
end

if ARGV.size == 2
	conll_prec ARGV[0], ARGV[1]
else
	conll_prec ARGV[0], ARGV[1], ARGV[2]
end
