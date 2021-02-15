var express = require('express');
var app = express();
var http = require('http').Server(app); //1
var io = require('socket.io')(http);
var bodyParser = require('body-parser');
var session = require('express-session');
var path = require("path");
var favicon = require("serve-favicon");

var fs = require("fs");
const { strict } = require('assert');

app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);

var server = http.listen(3000, function(){
 console.log("Express server has started on port 3000")
});

app.use(express.static('public'));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded());

app.use(favicon(path.join(__dirname, "public", "chatbot.png")));

app.use(session({
 secret: '@#@$MYSIGN#@$#$',
 resave: false,
 saveUninitialized: true
}));

var count=1;
io.on('connection', function(socket){ //3
  console.log('connected: ', socket.id);  //3-1

  socket.on('disconnect', function(){ //3-2
    console.log('user disconnected: ', socket.id);
  });

  socket.on('chat question', function(question){ //3-3
    console.log(question);
    //질문이 전송되었다.
    console.log("Socket ID = " + socket.id);
    //질문을 question_socket.id 파일 이름으로 저장한다.
    //먼저 지정한 단어가 있는 질문에 대해서는 그냥 대답한다.
    if(question.includes("한국IT") &&  question.includes("장점")) {
      socket.emit('chat answer', "1998년에 개설한 국내 유일 IT특성화 교육기관으로 실용주의 교육을 기본으로 기초학기와 방학기간 동안의 심화프로젝트학기, 프로젝트학습을 통한 협업능력과 문제해결능력, 창의력과 소통능력을 겸비한 4차산업혁명 리더를 양성하고 있습니다");
    }
    else if(question.includes("한국IT")){
      socket.emit('chat answer', "한국IT에는 현장경험이 풍부한 우수한 교수진이 있습니다. 한국IT에서는 산업현장과 똑같은 시설ㆍ장비를 갖추고, IT업체에서 일하듯이 실습을 합니다. 한국IT는 많은 IT업체와 산학협력을 맺고 있으며, 이미 많은 졸업생들이 IT업체에 진출해 있습니다. 한국IT가 위치한 양재동을 서울시와 서초구가 R&CD 특구로 지정하여 미국의 실리콘벨리처럼 키워가기로 했습니다. 한국IT에서는 제4차 산업혁명시대에 더욱 빛나는 전공분야를 선택할 수 있습니다.");
    }
    else if(question.includes("게임") &&  question.includes("학과")){
      socket.emit('chat answer', "게임기획 / 게임인공지능프로그래밍 / 게임아트디자인과정으로 나뉘어 현장 실무를 위한 전공세분화를 통해 풀버전의 게임을 개발하고 있습니다. 게임 완성작은 교내 프로젝트 경진대회 및 G-STAR, GGC에 출품하여 수상은 물론 경력까지 함께 쌓으며 경력 같은 신입으로 취업하게 됩니다. VR, AR개발센터와 모션캡쳐장비 등 첨단 교육시설과 담임 교수제를 통하여 대학에서 경험하기 힘든 섬세한 교육시스템으로 게임 전문가로 거듭나게 됩니다. ");
    }
    else if(question.includes("디자인") &&  question.includes("학과")){
      socket.emit('chat answer', "디지털 시각디자인 / 일러스트 / 웹툰 / 멀티미디어 / 디지털애니메이션과정으로 나뉘어 현장 실무를 위한 전공세분화를 통해 고퀄리티의 포트폴리오를 제작하고 있습니다. 제작된 포트폴리오는 교내 프로젝트 경진대회 및 국내 3대 페어(캐릭터일러스트디자인)에 출품하여 수상은 물론 경력까지 함께 쌓으며 경력 같은 신입으로 취업하게 됩니다. 만화도서관과 태블릿 실습실, 프로젝트실 등 차별화된 교육시설과 디자인업계 경력 10년 이상의 교수님들의 1:1 진로지도 및 학습설계로 일반대학과 차별화되는 디자인전문가로 거듭나게 됩니다. ");
    }
    else if(question.includes("정보보안") &&  question.includes("학과")){
      socket.emit('chat answer', "한국IT의 정보보안계열은 융합보안 / 정보보안 / 디지털포렌식 / 해킹 과정으로 나뉘어 현장 실무를 위한 전공세분화를 통해 최고의 기술력과 윤리의식을 갖춘 정보보안전문가를 양성하고 있습니다. 특히 전문 분야별 프로젝트 팀 구성을 기반으로 사전 토의(Project Build Up Camp)를 거쳐 실제 기업과 같은 프로세스로 프로젝트 작품을 개발 하여 교내 프로젝트 경진대회 참가는 물론 외부출품을 통해 경력 같은 신입으로 취업하게 됩니다. 또한 학년별 수준에 맞추어 방학 중 4주간 자체적으로 진행하는 미니 경진대회인 KITIS CONTEST를 통하여 실무를 즉시 활용하여 실력을 쌓고 있습니다. 디지털포렌식센터와 보안관제센터, 프로젝트실 등 현장과 동일한 첨단 교육시설과 보안업계 경력 10년 이상의 교수님들의 1:1 진로지도 및 학습설계로 일반대학과 차별화되는 보안전문가로 거듭나게 됩니다.");
    }
    else if(question.includes("인공지능") &&  question.includes("학과") ){
      socket.emit('chat answer', "한국IT의 인공지능학과는 컴퓨터공학 / 소프트웨어 / 로봇,드론 / 빅데이터, 인공지능 / 사물인터넷 과정으로 나뉘어 현장 실무를 위한 전공세분화를 통해 취업중심의 교육을 도모하고 있습니다. 특히 111제도로 1사람이 1년에 1개의 프로젝트를 개발하여 교내 프로젝트 경진대회는 물론 외부 출품을 통해 경력 같은 신입으로 취업하게 됩니다. 또한 하나의 프로젝트 작품을 목표로 전 계열 학생들이 기획 / 설계 / 제작하여 상용화 수준의 작품을 개발하는 창의인재 캠프를 통해 융복합 프로젝트 결과물을 제작하고 있습니다. 현장 경력 10년 이상의 교수님의 1:1 진로지도 및 학습설계로 일반 대학과 차별화 되는 IT전문가로 거듭나게 됩니다.");
    }
    else if(question.includes("취업")){
      socket.emit('chat answer', "개인별 맞춤형 취업지원이 가능한 한국IT취업지원센터를 통해 업체 발굴, 다양한 취업정보 제공, 취업 컨설팅 등 학생 한 사람 한사람에게 맞춰진 취업지원을 돕고 있습니다. ");
    }
    else if(question.includes("등록금")){
      socket.emit('chat answer', "홈페이지 ‘신입생 등록금 조회하기’ 팝업 메뉴에서 등록금 확인이 가능합니다.");
    }
    else if(question.includes("수업") &&  question.includes("시간")){
      socket.emit('chat answer', "본교는 주26시간가량 수업하고 있으며, 매 학기마다 시간표가 짜져서 나옵니다. 학과마다 상이하지만 주1회공강, 오후수업이 있으며 보통 9시부터 18시 사이에 수업이 진행됩니다. ");
    }
    else if(question.includes("선행") &&  question.includes("학습")){
      socket.emit('chat answer', "선행학습이란 예비 신입생(합격생)을 대상으로 3월 입학 전까지 전공 관련 기초 지식을 자율적으로 학습하도록 제공하는 Self-Learning시스템입니다. 입학안내-선행학습 탭을 클릭하여 무료로 수강할 수 있습니다.");
    }
    else if(question.includes("심화") &&  question.includes("프로젝트")){
      socket.emit('chat answer', "심화프로젝트학기는 여름방학 8주, 겨울방학 10주 과정을 기본으로 합니다. 각 학과마다 상이하나 자격증 취득반이 개설되며, 자기주도적 수업을 통하여 또 하나의 프로젝트 결과물을 만들어내기도 합니다. 이는 성적증명서에 표기되어 취업 시 매우 유리합니다.");
    }
    else if(question.includes("경쟁률") ){
      socket.emit('chat answer', "각 계열 경쟁률은 주단위로 변동되기 때문에 자세한 안내는 어려우나, 보통 3:1정도로 나타나고 있습니다.");
    }
    else if(question.includes("면접") &&  question.includes("준비물")){
      socket.emit('chat answer', "면접 시 준비물은 증명사진과 학과에 대한 관심과 열정입니다. 세부 준비 내역은 원서접수 후 담당 입학처 선생님께서 상세 안내해줍니다.");
    }
    else if(question.includes("면접") &&  question.includes("복장")){
      socket.emit('chat answer', "면접복장에는 제한이 없으나, 첫인상과 면접분위기를 위하여 단정한 의상 착용하시면 됩니다.");
    }
    else if(question.includes("기초") &&  question.includes("입학")){
      socket.emit('chat answer', "본교는 완벽한 인재가 아닌 배우고자 하는 의지와 열정을 토대로 학생을 선발하고 있습니다. 합격 후 전공기초를 쌓고 싶으시다면 본교에서 제공되는 선행학습을 이용하시면 됩니다.");
    }
    else if(question.includes("면접")){
      socket.emit('chat answer', "면접은 개인의 역량에 따라 달라집니다. 원서 접수 후 세부 면접 준비는 입학처 선생님께서 안내해주실 예정입니다.");
    }
    else if(question.includes("접수") &&  question.includes("기간")){
      socket.emit('chat answer', "접수기간은 한국IT전문학교 홈페이지에 기재되어있습니다. 확인이 어려운 경우 입학처 02 578 5551로 문의바랍니다.");
    }
    else if(question.includes("동아리") ){
      socket.emit('chat answer', "일반 대학과 마찬가지로 동아리가 개설되어 있어 정규 과정 외에 팀원들과의 스터디, 자율 토론 등을 통해 전공에 대한 깊이 있는 학습을 할 수 있게 됩니다.");
    }
    else if(question.includes("축제") ){
      socket.emit('chat answer', "본교는 축제가 없습니다. 하지만 일반 대학의 축제기간동안 프로젝트 경진대회를 열어 학생들이 1년간 제작한 작품을 전시 및 발표하여 피드백 갖는 시간을 갖고 있습니다. ");
    }
    else if(question.includes("기숙사") ){
      socket.emit('chat answer', "교내 남학생 생활관과 남학생과 여학생 모두 수용 가능한 교외생활관이 제공되고 있습니다. 자세한 금액과 컨디션은 학생서비스-숙소안내를 이용해주시기 바랍니다.");
    }
    else if(question.includes("식당") ){
      socket.emit('chat answer', "학생 식당은 학생들의 요청에 따라 운영하지 않고 있습니다. 하지만 양재관 내에 CU, 다산관 근처 GS25가 운영되고 있으며, 도보 5분 거리에 식당가가 있어 함께 이용하실 수 있습니다.");
    }
    else if(question.includes("편입")){
      socket.emit('chat answer', "본교 졸업 시 최소 2년제 전문학사학위가 인정되며 학점에 따라 4년제 학사학위 취득도 가능합니다. 4년제 대학교 3학년으로 일반편입 혹은 학사편입 지원이 가능하며 졸업 후 바로 대학원 진학도 가능합니다.");
    }
    else if(question.includes("불합격") &&  question.includes("지원")){
      socket.emit('chat answer', "같은 계열로는 재지원이 불가하나 다른 계열로 지원은 가능합니다. 같은 계열로 지원하고자 하면 다음년도에 지원해주세요.");
    }
    else if(question.includes("고2") &&  question.includes("지원")){
      socket.emit('chat answer', "검정고시 합격을 통해 고등학교 졸업 이상의 학력을 보유하고 있으면 지원 가능합니다.");
    }
    else if(question.includes("진로") &&  question.includes("체험")){
      socket.emit('chat answer', "매 달 셋째 주 토요일에 진로 체험 진행합니다.");
    }
    else if(question.includes("수업") &&  question.includes("비대면")){
      socket.emit('chat answer', "교양이나 이론으로 들을 수 있는 과목은 비대면으로 진행하나 실무중심으로 이루어져 있는 전공과목들은 대면과 비대면으로 동시에 수업합니다. 비대면과 대면 수업을 병행하여 본인이 선택할 수 있지만 실무위주의 수업이므로 대면수업을 권장합니다.");
    }
    else if(question.includes("면접") &&  question.includes("질문")){
      socket.emit('chat answer', "각 과마다 질문이 다릅니다. 공통적으로 준비해야 할 점은 간단한 자기소개와 지원동기이며 면접예상질문지는 면접 전 공개 예정입니다.");
    }
    else if(question.includes("나이") &&  question.includes("제한")){
      socket.emit('chat answer', "본교는 나이제한이 있으며 취업 가능한 선에서 받고 있습니다. 자세한 안내는 입학처 02 578 5551로 문의주시기 바랍니다.");
    }
    else if(question.includes("이중") &&  question.includes("학적")){
      socket.emit('chat answer', "본교는 학점은행제로 같은 학위증을 취득할 수 있는 곳이므로 이중학적은 불가능합니다. 타 대학을 자퇴하거나 졸업하고 나서 입학 가능합니다.");
    }
    else if(question.includes("학사") &&  question.includes("학위")){
      socket.emit('chat answer', "3년 교육과정을 통해 교육부장관 명의 앞으로 4년제 일반학사학위증을 취득할 수 있으며 이는 타 대학과 같은 학위증입니다. 단, 디자인계열은 2년 과정으로 전문학사를 취득할 수 있습니다.");
    }
    else if(question.includes("학자금") &&  question.includes("대출") &&  question.includes("방법")){
      socket.emit('chat answer', "①공식홈페이지-학생서비스-학자금대출에서 상세내용 확인하시기 바랍니다. \n ②만19세이상 신청 가능 \n ③본인 명의 휴대전화 필요 \n ④금리 3.5%(보증료율 0.5%, 포함시 4.0%) \n ※학점은행제 수강증명서는 교학처에서 발급 \n ※학점인정 증명서는 국가평생교육진흥원 학점은행 홈페이지에서 온라인 발급 \n ※상세문의 : 서민금융지원센터 1397");
    }
    else if(question.includes("학자금") &&  question.includes("대출")){
      socket.emit('chat answer', "학자금대출은 불가하나 본교 학생으로 대출받을 수 있는 은행지점으로 안내 받을 수 있습니다.");
    }
    else if(question.includes("군인") &&  question.includes("접수")){
      socket.emit('chat answer', "입학 전까지 전역이 완료된 상태면 군인도 접수 됩니다.");
    }
    else if(question.includes("한국IT") &&  question.includes("소개")){
      socket.emit('chat answer', "본교는 23년간 IT분야만 집중하고 있으며 IT분야 인력양성, 산업현장에 공급 발전을 기여하기 위해 힘쓰고 있습니다. IT특성화 교육기관이라는 명성에 어울리게 최신 업계 추세에 맞는 실무 교육과 다양한 장비를 활용하여 교육하고 있으며 실무에 특화된 프로젝트식 수업, 기업형 프로젝트를 통한 실무형 인재를 배출하고자 합니다.");
    }
    else if(question.includes("학점") &&  question.includes("은행")){
      socket.emit('chat answer', "학점은행제는 학점인정 등에 관한 볍률에 의거하여 학교 밖에서 이루어지는 다양한 형태의 학습과 자격을 학점으로 인정하고, 동등한 학사 학위증을 취득할 수 있습니다. 학사 학위를 취득할 수 있는 전문교육기관이며 본교에서 학점을 취득하면 교육부장관명의의 학위가 수여됩니다. 이는 4년제 대학에서 취득한 학위와 동일하게 국내·외 대학원에 진학할 수 있습니다.");
    }
    else if(question.includes("한국IT") &&  question.includes("강점")){
      socket.emit('chat answer', "본교는 심화 프로젝트 학기를 통해 자기주도 학습 능력과 문제해결능력을 키우고 있습니다. 이는 실제 현장에서 일하듯이 프로젝트를 진행하기 때문에 4차 산업혁명 시대에 큰 강점이 될 수 있습니다. 실무중심과 취업을 목적으로 커리큘럼이 구성되어 있어 실무중심의 학업수준을 향상시킬 수 있으며 학사학위 취득기간 단축으로 시간과 교육비를 절감할 수 있습니다.");
    }
    else if(question.includes("취업") &&  question.includes("졸업")){
      socket.emit('chat answer', "본교는 졸업인증제를 통해 인성교육과 프로젝트 수행능력, 소통능력, 협업능력, 문제해결능력, 창의력 등을 모두 다 겸임한 교육을 진행하고 있기 때문에 우수한 인재를 양성할 수 있습니다. 또한 현장 전문가를 통한 실무특강을 병행하고 있기 때문에 전공 학위취득으로 업계에서 ‘경력같은 신입’으로 인정받을 수 있습니다.");
    }
    else if(question.includes("콘텐츠") &&  question.includes("전망")){
      socket.emit('chat answer', "지금 트렌드는 하드웨어 보다 소프트웨어로 콘텐츠의 비중이 넘어가고 있고 지식재산권(IPR)의 중요성이 커지고 있습니다. 그에 대응하여 더 높은 수준의 콘텐츠를 개발하고 학생들의 수업환경을 강화하기 위해 여러 노력을 하고 있습니다. ");
    }
    else if(question.includes("프로젝트") &&  question.includes("효과")){
      socket.emit('chat answer', "산업현장에서 실무경험을 쌓듯이 경력자와 같은 졸업생을 배출합니다. 앞서 말한 것과 같이 ‘경력같은 신입’으로 보다 현장에 적응하기 쉽고 우수한 인재를 양성하고 있습니다.");
    }
    else if(question.includes("자퇴")){
      socket.emit('chat answer', "① 대상자 : 부득이한 사유에 의해 학업을 포기하고자 하는 자 \n ② 신청기간 : 제한없음 \n ③ 복적신청 : 자퇴자는 1회에 한해 복학할 수 있음 \n ※제출서류 : 자퇴원, 보호자 동의서, 보호자 통장사본(환불금액 있을 경우) \n ※반환금액 : 자퇴원 작성일자는 반환사유 발생일로 인정되며, 반환사유 발생일에 따라 반환금액이 결정됩니다. 반환에 대한 규정은 국가법령에 의거하여 진행됩니다. \n ※관련법령 : 평가인정 학습과정 운영에 관한 규정(제4조제2항 관련)에 따라 자퇴일이 개강일 기준 1/6까지는 총수업료의 5/6, 1/6이상부터 1/3미만까지는 총수업료의 2/3, 1/3이상부터 1/2미만까지는 총수업료의 1/2, 1/2이상 경과시는 반환금액이 없다.");
    }
    else if(question.includes("학적") &&  question.includes("변경") &&  question.includes("신청")){
      socket.emit('chat answer', "①지도교수님과 상담 (방문/통화) \n ②학적변경 서류 작성 \n ③지도교수님께 서류 제출 (방문이 원칙이지만 사정에 따라 팩스 또는 이메일 제출 가능) \n ④학적승인 \n ※더 자세한 사항 및 서류양식 다운로드는 공식홈페이지-학사안내-학적 메뉴에서 확인해주시기 바랍니다.");
    }
    else if(question.includes("학적") &&  question.includes("변경")){
      socket.emit('chat answer', "①지도교수님과 상담 (방문/통화) \n ②학적변경 서류 작성 \n ③지도교수님께 서류 제출 (방문이 원칙이지만 사정에 따라 팩스 또는 이메일 제출 가능) \n ④학적승인 \n ※더 자세한 사항 및 서류양식 다운로드는 공식홈페이지-학사안내-학적 메뉴에서 확인해주시기 바랍니다.");
    }
    else if(question.includes("학적") &&  question.includes("변경")){
      socket.emit('chat answer', "학적 변경 관련 서류양식은 지도교수님, 교학처에서 받으시거나 학교 공식홈페이지-학사안내-학적에서 다운로드 가능합니다.");
    }
    else if(question.includes("일반") &&  question.includes("휴학")){
      socket.emit('chat answer', "① 대상자 : 등록금을 납입한 자 \n ② 제외자 : 신입생이나 편입생의 첫 학기 \n ③ 신청기간 : 전체 수강 기간 중 개강일로부터 1/3일 이내  \n ④ 휴학기간 : 휴학원 작성일 부터 2학기(1년) 휴학  \n ⑤ 일반휴학은 최대 2 번까지 가능 \n ⑥ 수업료 이월 : 복학학기로 이월  \n ※제출서류 : 휴학원, 보호자 동의서 \n ※이미 일반휴학을 한 이후 입대하는 경우, 반드시 다음학기에 군복무확인서를 첨부하여 군휴학을 신청하여야 합니다. (미제출 시 제적 처리됩니다)");
    }
    else if(question.includes("군") &&  question.includes("휴학")){
      socket.emit('chat answer', "① 대상자 : 입영통지서를 배부 받은 자  \n ② 신청기간 : 입영일자가 전체 수강기간의 2/3 이전인 경우 \n  ※ 전체 수강기간의 2/3 초과 시는 중간고사 성적대비 최종 성적이 부여되며, 복학 시기는 성적이 부여, 다음 학기에 복학. \n ③ 휴학기간 : 휴학원 작성일 부터 군복무기간을 고려해서 복학학기 설정 \n ④ 수업료 이월 : 복학학기로 이월 (성적이 부여되는 경우 이월되지 않습니다) \n ※제출서류 : 휴학원, 입영통지서 사본");
    }
    else if(question.includes("가사") &&  question.includes("휴학")){
      socket.emit('chat answer', "① 대상자 : 기초생활수급자, 차상위계층, 한부모가족 \n ② 휴학기간 : 휴학원 작성 다음날 ~1년 휴학 \n ※ 신청 횟수 : 대상자 유지기간 동안 제한 없음 \n ※ 제출서류 : 휴학원, 기초생활수급자-수급자증명서, 차상위계층-차상위계층증명서, 한부모가족-한부모가족증명서");
    }
    else if(question.includes("질병") &&  question.includes("휴학")){
      socket.emit('chat answer', "① 대상자 : 4주 이상의 진단을 받은 자  \n ② 휴학기간 : 휴학원 작성일 부터 2학기(1년) 휴학  \n ※ 제출서류 : 휴학원, 4주 이상의 진단서");
    }
    else if(question.includes("복학")){
      socket.emit('chat answer', "① 대상자 : 휴학자, 자퇴자, 제적자 (징계대상 학생일 경우 제외)  \n ② 신청기간 : 개강 2주 전까지  \n ※ 제출서류 : 복학원 \n ※ 복학 학기에 자퇴 시는 휴학일과 자퇴일 중 최대 수강일을 기준으로 반환 처리됩니다.");
    }
    else if(question.includes("유급")){
      socket.emit('chat answer', "① 대상자 : 이전 학기 재수강을 원하는 자  \n ② 신청기간 : 개강 2주 전까지 \n ※ 제출서류 : 유급신청서");
    }
    else if(question.includes("학과") &&  question.includes("변경")){
      socket.emit('chat answer', "① 대상자 : 재학생, 복학 신청자  \n ② 신청기간 : 개강 2주 전까지  \n ※ 제출서류 : 학과변경 신청서 \n ※ 변경하고자 하는 학과의 정원을 초과하지 않는 범위 내에서만 변경 가능합니다.");
    }
    else if(question.includes("증명") &&  question.includes("서류")){
      socket.emit('chat answer', "①정부24 : 정부24 사이트를 통해 인터넷으로 발급 신청 후 주민센터 방문하여 수령 \n ②인터넷 발급 : 공식홈페이지-학생서비스-증명서발급-인터넷발급 홈페이지 접속하여 온라인 발급 \n ③교학처 방문 : 양재관 2층 교학처에 방문");
    }
    else if(question.includes("학점") &&  question.includes("인정")){
      socket.emit('chat answer', "①국가평생교육진흥원 학점은행 홈페이지 로그인-증명서 신청-학점인정 증명서 발급 \n https://www.cb.or.kr/creditbank/stuHelp/nStuHelp2_1.do");
    }
    else if(question.includes("학위") &&  question.includes("증명")){
      socket.emit('chat answer', "①국가평생교육진흥원 학점은행 홈페이지 로그인-증명서 신청-학위증명서 발급 \n https://www.cb.or.kr/creditbank/stuHelp/nStuHelp2_1.do \n ※ 학위수여 예정증명서는 학사 100학점, 전문학사 40학점 이상 인정되어있는 경우 접속가능함 \n ※ 총학점 요건 충족시 신청가능 (타전공 학습자는 제외) \n ※ 미인정 학점도 직접 입력 가능. 총학점 요건 미충족 시 발급 불가함");
    }
    else if(question.includes("등록금") &&  question.includes("고지서")){
      socket.emit('chat answer', "학사종합정보시스템에 기재된 보호자 주소로 등록금 납입기간 시작 1~2주전 일괄 우편 발송됩니다. 거주지 변경되거나 오기입 하셨을 경우 교학처로 연락주시기 바랍니다. 또, 학사종합정보시스템에서 온라인으로 확인 및 발급도 가능하오니 참고 하시기 바랍니다");
    }
    else if(question.includes("등록금") &&  question.includes("분할") &&  question.includes("납부")){
      socket.emit('chat answer', "등록금 분납을 지도교수님께 문의하시어 신청해보시기 바랍니다. 등록금 분납이란 개강 전 10만원을 납부하고 개강 후 2.5주 이내 남은 차액을 납부하는 제도입니다. \n ※제출서류 : 등록금분납원");
    }
    else if(question.includes("등록금") &&  question.includes("납부")){
      socket.emit('chat answer', "①등록금고지서의 가상계좌에 현금 납부  \n ②방문하여 카드 결제 (카드+현금, 2개 카드 분할납부 가능) \n ※평일 오전 9시~오후 6시 (점심시간 12시~1시) 양재관 7층 회계팀 방문 \n ※방문전, 미리 전화하여 무이자 할부 카드 확인 필요(변동 있음) \n ※공휴일,주말 휴무");
    }
    else if(question.includes("연말") &&  question.includes("정산")){
      socket.emit('chat answer', "보호자가 매년 1월중 연말정산 시 홈택스 연말정산간소화 서비스에서 확인 가능 \n ※등록금 납부 후 휴학하는 경우 등록금이 실제 사용되는 복학년도에 연말정산을 받으실 수 있습니다.");
    }
    else if(question.includes("장학금") &&  question.includes("신청")){
      socket.emit('chat answer', "①공식홈페이지 공지사항 및 학교 내 게시판 공지 내용 확인 (개강전 공지)  \n ②기한 내 증빙서류 제출(개강 후 2주 이내 신청, 기한 초과시 신청 불가) \n ※성적장학생은 각 계열(학과)에서 석차에 따라 배정 \n ※장학규정 제9조(장학생의 자격제한)에 따라 졸업인증제 규정의 제3조 3항을 충족하지 못한 학생, 휴학생 및 제적생, 직전 학기에 유기정학 이상의 징계 또는 형사 처벌을 받은 학생, \n 보훈장학금/근로장학금/임원장학금/교육비지원 장학금을 제외한 모든 장학금은 F학점이 있는 경우 대상자에서 제외됩니다.");
    }
    else if(question.includes("장학금") ){
      socket.emit('chat answer', "교내 장학금 안내는 학교 홈페이지 학사안내-장학안내 혹은 책자 맨 뒷장을 참고해주시면 감사하겠습니다.");
    }
    else if(question.includes("학점") &&  question.includes("신청")){
      socket.emit('chat answer', "①본교 및 강남직업전문학교에서 수강한 과목은 해당 학기 종료 후 교학처에서 일괄 학점신청 합니다. 단, 개인신청이 되어 있을 경우 본교에서 처리불가 및 신청수수료에 대한 반환은 없습니다. \n ②기타 타기관에서 수강한 과목은 개인이 직접 학점신청 및 결제를 완료해야 합니다. 개인신청 방법은 국가평생교육진흥원 학점은행제 홈페이지에서 온라인 신청하실 수 있습니다. \n ③온라인 학점신청 기간은 매해 소폭 변경되므로, 교학처 혹은 국가평생교육진흥원 학점은행 홈페이지를 통해 확인 바랍니다.");
    }
    else if(question.includes("평생") &&  question.includes("교육")){
      socket.emit('chat answer', "①평생교육바우처란, 학습자가 본인의 학습 요구에 따라 자율적으로 학습 활동을 결정하고 참여할 수 있도록 정부가 제공하는 평생교육 이용권입니다. \n ②지원금액 : 연간 35만원  \n ③신청자격 : 만 19세 이상 성인 중 기초생활수급자, 차상위계층, 기준 중위소득 65% 이하인 가구의 구성원. 단, 1인 가구의 경우 기준 중위소득 120% 이하 \n ④신청방법 및 자세한 내용은 평생교육바우처 홈페이지-사업안내-바우처 이용자 참고 \n ⑤카드 발급 완료 후 사용 방법은 교학처(02-578-2200)로 문의");
    }
    else if(question.includes("주소") ||  question.includes("위치")){
      socket.emit('chat answer', "서울특별시 서초구 바우뫼로 87 (양재동 145-5)");
    }
    else if(question.includes("찾아") ||  question.includes("가는길")){
      socket.emit('chat answer', "양재역 11번출구에서 541, 18, 18-1 번 버스를 타고 양재초등학교앞에서 하차하시면 50미터 앞에 한국IT직업전문학교가 있습니다.");
    }
    else if(question.includes("인공지능") ||  question.includes("AI")){
      socket.emit('chat answer', "인공지능(人工知能, 영어: artificial Intelligence, AI)은 인간의 학습능력, 추론능력, 지각능력, 논증능력, 자연언어의 이해능력 등을 인공적으로 구현한 컴퓨터 프로그램 또는 이를 포함한 컴퓨터 시스템이다.");
    }
    else if(question.includes("챗봇") ||  question.includes("chatbot")){
      socket.emit('chat answer', "챗봇(chatbot) 혹은 채터봇(Chatterbot)은 음성이나 문자를 통한 인간과의 대화를 통해서 특정한 작업을 수행하도록 제작된 컴퓨터 프로그램이다. 토크봇(talkbot), 채터박스(chatterbox) 혹은 그냥 봇(bot)라고도 한다.");
    }
    else if(question.includes("코로나")){
      socket.emit('chat answer', "코로나바이러스감염증-19(코로나19)는 과거에 발견되지 않았던 새로운 코로나바이러스인 SARS-CoV-2에 의해 발생하는 호흡기 감염병입니다. 이 바이러스에 감염되면 무증상부터 중증에 이르기까지 다양한 임상증상이 나타날 수 있습니다. 이 새로운 바이러스와 질병은 2019년 12월 중국 우한에서 처음 보고되었고, 현재 전 세계에 확산되었습니다.");
    }
    else {
      // answer_socket.id 파일을 읽어서 클라이언트에게 보낸다.
      var filename = './qna/question_' + socket.id;
      fs.writeFileSync(filename, question, 'utf8');
      // answer_socket.id 파일이 생성 될떄 까지 기다린다.
      var ansfilename = './qna/answer_' + socket.id;
      var bFound = false;

      //1초 간격으로 3초 기다린다.
      var waitTill = new Date(new Date().getTime() + 500);
      while(waitTill > new Date()){}
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (fs.existsSync(ansfilename)) {
        bFound = true;
      }
      //있으면 파일에서 읽어서 전송한다.
      if (bFound) {
        console.log("answer founded = " + ansfilename);
        var answer = '';
        answer = fs.readFileSync(ansfilename, 'utf8');
        socket.emit('chat answer', answer);  //보낸 클라이언트에게만 메세지 전송
      }
      else {
        console.log("answer Not found = " + ansfilename);
        socket.emit('chat answer', '학교 대표전화 02-578-5551을 통해 문의해주세요.');  //보낸 클라이언트에게만 메세지 전송
      }
    }
    //io.emit('receive message', msg);    //전체에게 메세지 전송
  });
});

var router = require('./router/main')(app, fs);